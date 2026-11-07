"""全部使用假响应和假密钥，不联网、不消费额度；不是模型准确率评测。"""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import unittest
from unittest.mock import patch, MagicMock
from urllib.error import HTTPError, URLError

from lessons.creative_studio import deepseek_review as module
from lessons.creative_studio.promotion_review import load_materials


class DeepSeekReviewTests(unittest.TestCase):
    def setUp(self):
        self.rules, bundle = load_materials()
        self.case = bundle["cases"][0]
        self.claim = deepcopy(self.case["interpretation"])

    def envelope(self, content=None, finish="stop"):
        return {"choices": [{"finish_reason": finish, "message": {
            "content": json.dumps(self.claim) if content is None else content}}]}

    def test_payload_contains_only_headline_not_snapshot_or_rules(self):
        payload = module.build_payload(self.case["headline"])
        self.assertEqual(json.loads(payload["messages"][1]["content"]),
                         {"headline": self.case["headline"]})
        self.assertEqual(payload["max_tokens"], 512)
        self.assertEqual(payload["thinking"], {"type": "disabled"})

    def test_dry_run_never_calls_provider(self):
        with patch.object(module, "call_deepseek") as call, redirect_stdout(io.StringIO()):
            self.assertEqual(module.main([]), 0)
        call.assert_not_called()

    def test_live_calls_once_and_preserves_input(self):
        before = deepcopy(self.case)
        with patch.object(module, "call_deepseek", return_value=(self.envelope(), 0.1)) as call:
            with patch.dict(module.os.environ, {"DEEPSEEK_API_KEY": "fake-test-key"}), redirect_stdout(io.StringIO()) as output:
                self.assertEqual(module.main(["--live"]), 0)
        self.assertEqual(call.call_count, 1)
        self.assertIn("conflict_given_interpretation", output.getvalue())
        self.assertEqual(self.case, before)

    def test_live_response_replaces_old_interpretation(self):
        self.case["interpretation"]["claimed_total_cents"] = 6900
        result = module.review_response(self.case, self.rules, self.envelope())
        self.assertEqual(result["interpretation"]["claimed_total_cents"], 5900)
        self.assertEqual(self.case["interpretation"]["claimed_total_cents"], 6900)

    def test_incomplete_or_empty_or_bad_response(self):
        for envelope in ({}, {"choices": []}, self.envelope(finish="length"),
                         self.envelope(content=""), self.envelope(content="```json\n{}\n```")):
            with self.subTest(envelope=envelope), self.assertRaises(ValueError):
                module.parse_interpretation(envelope, self.case["headline"])

    def test_duplicate_or_extra_keys(self):
        for content in ('{"quote":"a","quote":"b"}', json.dumps(dict(self.claim, approved=True))):
            with self.assertRaises(ValueError):
                module.parse_interpretation(self.envelope(content), self.case["headline"])

    def test_wrong_quote_or_types(self):
        for field, value in (("quote", "伪引用"), ("quantity", True), ("quantity", 0),
                             ("claimed_total_cents", "5900"), ("claims_coupon_applied", 1),
                             ("price_basis", "unit_price")):
            claim = dict(self.claim, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                module.parse_interpretation(self.envelope(json.dumps(claim)), self.case["headline"])

    def test_unknown_values_go_to_confirmation(self):
        self.claim.update(quantity=None, price_basis="unclear", claimed_total_cents=None,
                          claims_coupon_applied=None)
        result = module.review_response(self.case, self.rules, self.envelope())
        self.assertEqual(result["review"]["status"], "needs_confirmation")

    def test_missing_key_stops_before_network(self):
        with patch.object(module.request, "build_opener") as opener:
            with self.assertRaises(module.CallFailure):
                module.call_deepseek({}, None)
        opener.assert_not_called()

    def test_transport_and_no_secret_in_payload(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(self.envelope()).encode()
        opener = MagicMock()
        opener.open.return_value = response
        with patch.object(module.request, "build_opener", return_value=opener):
            envelope, elapsed = module.call_deepseek(module.build_payload(self.case["headline"]), "fake-test-key")
        req = opener.open.call_args.args[0]
        self.assertEqual(req.full_url, module.API_URL)
        self.assertEqual(req.get_header("Authorization"), "Bearer fake-test-key")
        self.assertNotIn(b"fake-test-key", req.data)
        self.assertEqual(opener.open.call_count, 1)
        self.assertGreaterEqual(elapsed, 0)

    def test_http_and_network_errors_are_sanitized_without_retry(self):
        for exc in (HTTPError(module.API_URL, 401, "fake-secret", {}, None), URLError("fake-secret")):
            opener = MagicMock()
            opener.open.side_effect = exc
            with patch.object(module.request, "build_opener", return_value=opener):
                with self.assertRaises(module.CallFailure) as caught:
                    module.call_deepseek({}, "fake-test-key")
            self.assertNotIn("fake-secret", str(caught.exception))
            self.assertEqual(opener.open.call_count, 1)

    def test_redirect_is_disabled(self):
        self.assertIsNone(module.NoRedirect().redirect_request(None, None, 302, "", {}, "https://other.invalid"))


if __name__ == "__main__":
    unittest.main()
