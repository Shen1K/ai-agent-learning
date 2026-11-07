"""第二步：版本化规则 + 文案理解快照 + 确定性核算。

这不是自然语言审核模型。interpretation是教学快照，不是本脚本推理所得。
阅读顺序：main → review_case → calculate_order → coupon_discount。
金额统一用整数“分”，避免浮点数给边界比较带来误差。
"""

from copy import deepcopy
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_materials():
    """以代码所在目录定位材料，不受终端工作目录影响；仅读取。"""
    rules = json.loads((HERE / "promotion_rules_v1.json").read_text(encoding="utf-8"))
    cases = json.loads((HERE / "promotion_cases.json").read_text(encoding="utf-8"))
    return rules, cases


def require_integer(value, name, minimum=0):
    """输入约定检查，不是业务判断；True在Python中属于int子类，须排除。"""
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name}必须是大于等于{minimum}的整数")


def validate_rules(rules):
    """规则不完整就暂停，不能偷偷采用默认优惠条件。

    返回缺失字段列表；格式错误或本课不支持的策略则抛出ValueError。
    这里只支持一张券、不叠加、单一商品；并非通用促销规则引擎。
    """
    if not isinstance(rules, dict):
        raise ValueError("规则必须是对象")
    required = ("activity_id", "version", "product_id", "currency", "unit_price_cents",
                "threshold_cents", "discount_cents", "shipping_counts_toward_threshold",
                "max_coupons_per_order", "allow_stacking")
    missing = [name for name in required if rules.get(name) is None]
    if missing:
        return missing
    for name in ("activity_id", "version", "product_id"):
        if not isinstance(rules[name], str) or not rules[name].strip():
            raise ValueError(f"{name}必须是非空字符串")
    for name in ("unit_price_cents", "threshold_cents", "discount_cents"):
        require_integer(rules[name], name)
    if rules["discount_cents"] > rules["threshold_cents"]:
        raise ValueError("本课不支持优惠额大于门槛的规则，请先明确业务处理方式")
    if type(rules["shipping_counts_toward_threshold"]) is not bool:
        raise ValueError("运费是否计入门槛必须明确为true或false")
    if (rules["currency"] != "CNY" or rules["allow_stacking"] is not False
            or type(rules["max_coupons_per_order"]) is not int
            or rules["max_coupons_per_order"] != 1):
        raise ValueError("本课仅支持人民币、每单一张券、不叠加其他优惠")
    return []


def coupon_discount(eligible_cents, threshold_cents, discount_cents):
    """只回答门槛是否满足；eligible_cents由上游按规则计算。

    >=表示包含门槛本身：9900分可以用券，9899分不可以。
    此函数不知道运费或商品类别，不能单凭它正确就说整个业务正确。
    """
    for name, value in (("eligible_cents", eligible_cents),
                        ("threshold_cents", threshold_cents),
                        ("discount_cents", discount_cents)):
        require_integer(value, name)
    if discount_cents > threshold_cents:
        raise ValueError("优惠额不能大于本课门槛")
    if eligible_cents >= threshold_cents:
        return discount_cents
    return 0


def calculate_order(order, rules):
    """仅消费已确认的结构化订单与规则，不读取模型自报的正确价格。

    本课假定活动有效且用户持券；没有时间、库存或领券资格检查。
    返回中间金额，出错时能追踪是门槛金额、减免还是最终总价有问题。
    """
    if validate_rules(rules):
        raise ValueError("规则缺失，不能计算")
    if not isinstance(order, dict):
        raise ValueError("订单必须是对象")
    if order.get("product_id") != rules["product_id"]:
        raise ValueError("订单商品不属于本规则")
    require_integer(order.get("quantity"), "quantity", minimum=1)
    require_integer(order.get("shipping_cents"), "shipping_cents")
    if order.get("other_discounts_present") is not False:
        raise ValueError("需先确认无其他优惠，不能按本课公式直接计算")

    goods = rules["unit_price_cents"] * order["quantity"]
    shipping = order["shipping_cents"]
    eligible = goods
    if rules["shipping_counts_toward_threshold"]:
        eligible = goods + shipping
    discount = coupon_discount(eligible, rules["threshold_cents"], rules["discount_cents"])
    return {"goods_cents": goods, "shipping_cents": shipping,
            "eligible_cents": eligible, "discount_cents": discount,
            "total_cents": goods + shipping - discount}


def review_case(case, rules):
    """审核一个候选理解，不声称已验证自由文案的全部语义。

    引用存在只证明文字来源存在，不证明quantity等字段被正确理解。
    因此即使核算相符，也只返回consistent_given_interpretation，不能发布。
    """
    result = {"case_id": case["id"], "scope": "one_order_price_claim",
              "rule_version": rules.get("version"), "can_publish": False}
    try:
        missing = validate_rules(rules)
        if missing:
            return dict(result, status="needs_information", reason="规则字段缺失", missing=missing)
        if (case.get("activity_id") != rules["activity_id"]
                or case.get("rule_version") != rules["version"]):
            return dict(result, status="needs_confirmation", reason="活动或规则版本不匹配")
        calculation = calculate_order(case["order"], rules)
        result["calculation"] = calculation
        claim = case.get("interpretation")
        if not isinstance(claim, dict):
            return dict(result, status="needs_confirmation", reason="没有有效的文案理解结果")
        # 本课快照引用整条标题。精确相等检查不能替代语义核验。
        if not isinstance(case.get("headline"), str) or claim.get("quote") != case["headline"]:
            return dict(result, status="needs_confirmation", reason="引用与待审标题不一致")
        if claim.get("price_basis") != "order_total_including_shipping":
            return dict(result, status="needs_confirmation", reason="不是明确的含运费订单总价，不能直接比较")
        require_integer(claim.get("quantity"), "候选quantity", minimum=1)
        require_integer(claim.get("claimed_total_cents"), "候选claimed_total_cents")
        if type(claim.get("claims_coupon_applied")) is not bool:
            raise ValueError("候选必须明确是否声称已用券")
        if claim["quantity"] != case["order"]["quantity"]:
            return dict(result, status="needs_confirmation", reason="候选理解的数量与订单情境不一致")

        reasons = []
        if claim["claims_coupon_applied"] and calculation["discount_cents"] == 0:
            reasons.append("文案声称已用券，但订单未获得券减免")
        if claim["claimed_total_cents"] != calculation["total_cents"]:
            reasons.append("文案声称总价与规则核算总价不一致")
        result["quote"] = claim["quote"]
        if reasons:
            return dict(result, status="conflict_given_interpretation",
                        reasons=reasons, next_action="复核语义后退回修改")
        return dict(result, status="consistent_given_interpretation",
                    next_action="核对文案理解与遗漏主张，再进行其他审核")
    except (ValueError, KeyError, TypeError) as error:
        # 未知／不支持不等于通过；记录错误并转确认，而不是猜一个总价。
        return dict(result, status="needs_confirmation", reason=str(error))


def main():
    rules, bundle = load_materials()
    print("[1 规则] 教学虚构活动", rules["activity_id"], "版本", rules["version"])
    print("[2 文案理解] 重放教师编写的快照；没有模型API调用。")
    for case in bundle["cases"]:
        print("\n标题：", case["headline"])
        print("理解快照：", json.dumps(case["interpretation"], ensure_ascii=False))
        print("[3 核算与处理]", json.dumps(review_case(case, rules), ensure_ascii=False, indent=2))
    # 对照只改规则副本，原规则文件不变。不能默认为False后继续计算。
    incomplete = deepcopy(rules)
    incomplete["shipping_counts_toward_threshold"] = None
    print("\n[4 缺失规则对照] 运费是否计入门槛未明确")
    print(json.dumps(review_case(bundle["cases"][2], incomplete), ensure_ascii=False, indent=2))
    print("\n边界：以上只核算指定订单主张，不生成图片、不发布、不证明完整语义审核通过。")


if __name__ == "__main__":
    main()
