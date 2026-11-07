"""第一课：读取约定资料，检查后组装任务，不调用模型、不生成图片。

阅读顺序：main → load_fixture → validate_contract → assemble_task。
数据是课堂虚构资料，不是线上商品系统。本文件只用 Python 标准库。
"""

import argparse
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path


# __file__ 是本代码文件的位置。以它定位资料，就不依赖终端当前目录。
# 因此，从项目根目录、scripts 目录或其他目录运行，都能找到默认资料。
DEFAULT_FIXTURE = Path(__file__).resolve().with_name("lesson01_fixture.json")


def load_fixture(path):
    """读取文件：JSON 对象变为 dict（字典），null 变为 None。

    这里只读取，不改写源文件；文件不存在、JSON 格式错误会抛出异常，
    由 main 统一展示错误，不能把读取失败误报为“商品没有价格”。
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_contract(data):
    """检查本课消费的字段约定；格式错误与合法的资料缺失分开处理。

    ValueError 表示交接数据不符合约定，不能继续组装。
    价格字段存在但值为 None 则是合法的“未提供”，留给业务分支处理。
    这是教学用局部校验，不是完整商品 Schema，也不验证自由文案语义。
    """
    if not isinstance(data, dict):
        raise ValueError("资料根节点必须是JSON对象")
    for section in ("product", "request", "brand_rules", "creative_proposal"):
        if not isinstance(data.get(section), dict):
            raise ValueError(f"{section}必须是对象")

    # 方括号按键取值；product、request 等是同一份资料的不同部分。
    product = data["product"]
    request = data["request"]
    rules = data["brand_rules"]
    proposal = data["creative_proposal"]
    for name in ("id", "version", "brand", "name"):
        value = product.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"product.{name}必须是非空字符串")
    if request.get("product_id") != product["id"]:
        raise ValueError("任务商品编号与资料商品编号不一致")
    if not isinstance(rules.get("version"), str) or not rules["version"].strip():
        raise ValueError("品牌规则必须提供版本")

    # 本课只有一个固定规格，不能接收其他规格却偷偷按1024输出。
    for name, expected in (("width_px", 1024), ("height_px", 1024),
                           ("variant_count", 1)):
        if type(request.get(name)) is not int or request[name] != expected:
            raise ValueError(f"本课request.{name}只支持整数{expected}")
    if request.get("show_activity_price") is not True:
        raise ValueError("本课任务必须明确show_activity_price=true")

    # 将本课支持的策略写成明确检查，避免读取规则后实际忽略它。
    supported_rules = {
        "allow_unverified_product_claims": False,
        "price_source": "product.activity_price_cny",
        "product_visual_strategy": "reuse_approved_product_asset",
        "text_strategy": "render_text_separately",
        "publication_policy": "local_draft_only",
    }
    for name, expected in supported_rules.items():
        value = rules.get(name)
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"本课不支持brand_rules.{name}的当前设置")

    for name in ("origin", "headline", "scene", "background_description"):
        value = proposal.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"creative_proposal.{name}必须是非空字符串")
    if not isinstance(proposal.get("product_claims"), list):
        raise ValueError("creative_proposal.product_claims必须是列表")

    for name in ("activity_price_cny", "reference_image_path"):
        if name not in product:
            raise ValueError(f"product缺少{name}字段；未提供请明确写null")
    price = product["activity_price_cny"]
    if price is not None:
        # Python 中 bool 是 int 的子类，所以这里用 type 排除 True/False。
        # 不默默把字符串"69"、负数、无穷大或三位小数当作合法价格。
        if type(price) not in (int, float):
            raise ValueError("价格必须是数字或null，不能是字符串或布尔值")
        amount = Decimal(str(price))
        if not amount.is_finite() or amount < 0:
            raise ValueError("价格必须是有限的非负数")
        fractional_digits = format(amount, "f").partition(".")[2]
        if any(digit != "0" for digit in fractional_digits[2:]):
            raise ValueError("价格最多两位小数，不自动四舍五入业务价格")
    asset = product["reference_image_path"]
    if asset is not None and (not isinstance(asset, str) or not asset.strip()):
        raise ValueError("商品图片路径必须是非空字符串或null")


def assemble_task(data, asset_base):
    """组装新字典，不修改传入资料，也不沿用旧 execution_plan。

    asset_base 是相对素材路径的基准目录，本课使用资料文件所在目录。
    输出是任务草案，不是图片；候选文案与素材都还需要审核。
    """
    validate_contract(data)
    product = data["product"]
    request = data["request"]
    proposal = data["creative_proposal"]

    # 列表可保存多个问题。不能用 if/elif 让缺价格掩盖缺图片。
    blocking_issues = []
    price = product["activity_price_cny"]
    price_text = None
    if price is None:
        blocking_issues.append({
            "code": "missing_activity_price",
            "message": "活动价未提供，请负责价格的业务方补充；不能估价。",
        })
    else:
        # 0是明确提供的值，不等于None。本课允许0，是否允许免费是另一条业务规则。
        # Decimal按十进制处理显示，69显示为69，69.90显示为69.9，不改价格。
        amount = format(Decimal(str(price)), "f")
        if "." in amount:
            amount = amount.rstrip("0").rstrip(".")
        price_text = f"活动价 ¥{amount}"

    asset = product["reference_image_path"]
    resolved_asset = None
    if asset is None:
        blocking_issues.append({"code": "missing_product_asset",
                                "message": "商品图片未提供，请补充待确认的商品素材。"})
    else:
        # 这里只检查本地文件存在。不证明它可解码、是正确商品或已经获得授权。
        asset_path = (Path(asset_base) / asset).resolve()
        if not asset_path.is_file():
            blocking_issues.append({"code": "product_asset_not_found",
                                    "message": "指定商品素材不是现存文件，请核对路径。"})
        else:
            resolved_asset = str(asset_path)

    # 从模型候选中只选创意字段；不把候选整体合并到商品事实里。
    # 即使候选里夹带price_text=59，下面仍只从商品资料得到价格。
    creative_candidate = {}
    for name in ("origin", "headline", "scene", "background_description", "product_claims"):
        creative_candidate[name] = deepcopy(proposal[name])

    # 资料齐备也只是needs_review，不是ready或completed。
    # 本课没有审核器和渲染器，不能凭文件存在就承诺可以交付。
    return {
        "product_id": product["id"],
        "source_product_version": product["version"],
        "source_brand_rules_version": data["brand_rules"]["version"],
        "canvas": {"width_px": request["width_px"], "height_px": request["height_px"]},
        "overlay_text": {"brand": product["brand"], "product_name": product["name"],
                         "headline": proposal["headline"], "price_text": price_text},
        "creative_candidate": creative_candidate,
        "reference_image_path": resolved_asset,
        "status": "needs_input" if blocking_issues else "needs_review",
        "blocking_issues": blocking_issues,
        "review_required": ["创意不得暗示未证实的商品事实", "商品素材内容、格式与使用授权"],
        "can_render": False,
        "image_generation_executed": False,
        "publication_policy": "local_draft_only",
    }


def main():
    """演示有价／缺价对照；变体只改内存副本，不覆盖69元的原始资料。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE,
                        help="课堂JSON资料路径；默认取代码旁的示例")
    args = parser.parse_args()
    try:
        data = load_fixture(args.fixture)
        validate_contract(data)
        print(f"[1 读取] {args.fixture.resolve()}")
        print("[2 约定检查] 通过；旧execution_plan是历史示例，本次重新组装。")
        for label, missing_price in (("原始资料", False), ("仅将价格改为null", True)):
            sample = deepcopy(data)  # 深拷贝也复制内部字典，后续修改不污染原始资料。
            if missing_price:
                sample["product"]["activity_price_cny"] = None
            plan = assemble_task(sample, args.fixture.resolve().parent)
            print(f"\n[3 组装] {label}")
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            print("[4 执行边界] 仅输出任务草案，不调用模型、不生成、不发布。")
    except (OSError, UnicodeError, ValueError) as error:
        print(f"输入错误，已停止：{error}")
        return 1
    return 0  # 代表演示程序正常结束，不代表海报已完成。


# 直接运行本文件才启动main；测试导入函数时不会自动执行演示。
if __name__ == "__main__":
    raise SystemExit(main())
