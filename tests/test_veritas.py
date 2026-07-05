"""
Unit tests for Veritas's custom transaction types.

Covers stateless check_tx validation and full deliver_tx state-transition
logic (using an in-memory FakePlugin) for:
  - register_evaluator
  - submit_evaluation_request
  - submit_ai_verdict (including the atomic auto-finalize/auto-flag path)
"""

import pytest

from contract.contract import (
    Contract,
    key_for_evaluator,
    key_for_evaluation,
    key_for_reputation,
    marshal,
    unmarshal,
)
from contract.plugin import Config
from contract.error import PluginError
from contract.proto import (
    MessageRegisterEvaluator,
    MessageSubmitEvaluationRequest,
    MessageSubmitAiVerdict,
    Evaluator,
    Evaluation,
    Reputation,
    PluginStateReadRequest,
    PluginStateReadResponse,
    PluginStateWriteRequest,
    PluginStateWriteResponse,
    PluginReadResult,
    PluginStateEntry,
)

# Error codes (see contract/error.py)
CODE_INVALID_ADDRESS = 12
CODE_INVALID_AMOUNT = 13
CODE_INVALID_CONTENT_HASH = 15
CODE_INVALID_SCORE = 16
CODE_INVALID_MODEL_NAME = 17
CODE_INVALID_EVALUATION_ID = 18
CODE_EVALUATION_ALREADY_EXISTS = 19
CODE_EVALUATION_NOT_FOUND = 20
CODE_EVALUATION_NOT_PENDING = 21
CODE_EVALUATOR_NOT_REGISTERED = 22

ADMIN = b"a" * 20
SUBMITTER = b"s" * 20
EVALUATOR_1 = b"e" * 20
EVALUATOR_2 = b"f" * 20
EVALUATOR_3 = b"g" * 20
ADDR_SHORT = b"short"

CONTENT_HASH = "abc123hash"


class FakePlugin:
    """In-memory stand-in for the real FSM-backed Plugin, for testing deliver_tx logic."""

    def __init__(self):
        self.store = {}

    async def state_read(self, contract, request: PluginStateReadRequest) -> PluginStateReadResponse:
        results = []
        for key_read in request.keys:
            entries = []
            if key_read.key in self.store:
                entries.append(PluginStateEntry(value=self.store[key_read.key]))
            results.append(PluginReadResult(query_id=key_read.query_id, entries=entries))
        return PluginStateReadResponse(results=results)

    async def state_write(self, contract, request: PluginStateWriteRequest) -> PluginStateWriteResponse:
        for op in request.sets:
            self.store[op.key] = op.value
        for op in request.deletes:
            self.store.pop(op.key, None)
        return PluginStateWriteResponse()


@pytest.fixture
def config():
    return Config()


@pytest.fixture
def fake_plugin():
    return FakePlugin()


@pytest.fixture
def contract(config, fake_plugin):
    c = Contract(config=config)
    c.plugin = fake_plugin
    return c


async def register_evaluator(contract, evaluator_address=EVALUATOR_1, model_name="llama-3.3-70b"):
    msg = MessageRegisterEvaluator(
        admin_address=ADMIN, evaluator_address=evaluator_address, model_name=model_name
    )
    return await contract._deliver_message_register_evaluator(msg)


async def submit_evaluation(contract, required_verdicts=2, content_hash=CONTENT_HASH):
    msg = MessageSubmitEvaluationRequest(
        submitter_address=SUBMITTER,
        content_hash=content_hash,
        anomaly_score=0.1,
        required_verdicts=required_verdicts,
    )
    return await contract._deliver_message_submit_evaluation_request(msg)


async def submit_verdict(contract, evaluator_address, score, evaluation_id=CONTENT_HASH, justification="ok"):
    msg = MessageSubmitAiVerdict(
        evaluator_address=evaluator_address,
        evaluation_id=evaluation_id,
        score=score,
        justification=justification,
    )
    return await contract._deliver_message_submit_ai_verdict(msg)


class TestCheckMessageRegisterEvaluator:
    def test_valid(self, contract):
        msg = MessageRegisterEvaluator(admin_address=ADMIN, evaluator_address=EVALUATOR_1, model_name="llama-3.3-70b")
        result = contract._check_message_register_evaluator(msg)
        assert not result.HasField("error")
        assert list(result.authorized_signers) == [ADMIN]

    def test_invalid_evaluator_address(self, contract):
        msg = MessageRegisterEvaluator(admin_address=ADMIN, evaluator_address=ADDR_SHORT, model_name="llama-3.3-70b")
        with pytest.raises(PluginError) as exc:
            contract._check_message_register_evaluator(msg)
        assert exc.value.code == CODE_INVALID_ADDRESS

    def test_empty_model_name(self, contract):
        msg = MessageRegisterEvaluator(admin_address=ADMIN, evaluator_address=EVALUATOR_1, model_name="")
        with pytest.raises(PluginError) as exc:
            contract._check_message_register_evaluator(msg)
        assert exc.value.code == CODE_INVALID_MODEL_NAME


class TestCheckMessageSubmitEvaluationRequest:
    def test_valid(self, contract):
        msg = MessageSubmitEvaluationRequest(
            submitter_address=SUBMITTER, content_hash=CONTENT_HASH, anomaly_score=0.1, required_verdicts=2
        )
        result = contract._check_message_submit_evaluation_request(msg)
        assert not result.HasField("error")

    def test_empty_content_hash(self, contract):
        msg = MessageSubmitEvaluationRequest(
            submitter_address=SUBMITTER, content_hash="", anomaly_score=0.1, required_verdicts=2
        )
        with pytest.raises(PluginError) as exc:
            contract._check_message_submit_evaluation_request(msg)
        assert exc.value.code == CODE_INVALID_CONTENT_HASH

    def test_zero_required_verdicts(self, contract):
        msg = MessageSubmitEvaluationRequest(
            submitter_address=SUBMITTER, content_hash=CONTENT_HASH, anomaly_score=0.1, required_verdicts=0
        )
        with pytest.raises(PluginError) as exc:
            contract._check_message_submit_evaluation_request(msg)
        assert exc.value.code == CODE_INVALID_AMOUNT


class TestCheckMessageSubmitAiVerdict:
    def test_valid(self, contract):
        msg = MessageSubmitAiVerdict(evaluator_address=EVALUATOR_1, evaluation_id=CONTENT_HASH, score=80, justification="good")
        result = contract._check_message_submit_ai_verdict(msg)
        assert not result.HasField("error")

    def test_score_over_100(self, contract):
        msg = MessageSubmitAiVerdict(evaluator_address=EVALUATOR_1, evaluation_id=CONTENT_HASH, score=150, justification="bad")
        with pytest.raises(PluginError) as exc:
            contract._check_message_submit_ai_verdict(msg)
        assert exc.value.code == CODE_INVALID_SCORE

    def test_empty_evaluation_id(self, contract):
        msg = MessageSubmitAiVerdict(evaluator_address=EVALUATOR_1, evaluation_id="", score=80, justification="good")
        with pytest.raises(PluginError) as exc:
            contract._check_message_submit_ai_verdict(msg)
        assert exc.value.code == CODE_INVALID_EVALUATION_ID


@pytest.mark.asyncio
class TestDeliverRegisterEvaluator:
    async def test_writes_evaluator_record(self, contract, fake_plugin):
        result = await register_evaluator(contract)
        assert not result.HasField("error")

        stored = fake_plugin.store[key_for_evaluator(EVALUATOR_1)]
        evaluator = unmarshal(Evaluator, stored)
        assert evaluator.model_name == "llama-3.3-70b"
        assert evaluator.active is True


@pytest.mark.asyncio
class TestDeliverSubmitEvaluationRequest:
    async def test_creates_pending_evaluation(self, contract, fake_plugin):
        result = await submit_evaluation(contract)
        assert not result.HasField("error")

        stored = fake_plugin.store[key_for_evaluation(CONTENT_HASH)]
        evaluation = unmarshal(Evaluation, stored)
        assert evaluation.status == "pending"
        assert evaluation.required_verdicts == 2
        assert len(evaluation.verdicts) == 0

    async def test_duplicate_content_hash_rejected(self, contract):
        await submit_evaluation(contract)
        with pytest.raises(PluginError) as exc:
            await submit_evaluation(contract)
        assert exc.value.code == CODE_EVALUATION_ALREADY_EXISTS


@pytest.mark.asyncio
class TestDeliverSubmitAiVerdict:
    async def test_rejects_unregistered_evaluator(self, contract):
        await submit_evaluation(contract)
        with pytest.raises(PluginError) as exc:
            await submit_verdict(contract, EVALUATOR_1, 80)
        assert exc.value.code == CODE_EVALUATOR_NOT_REGISTERED

    async def test_rejects_missing_evaluation(self, contract):
        await register_evaluator(contract, EVALUATOR_1)
        with pytest.raises(PluginError) as exc:
            await submit_verdict(contract, EVALUATOR_1, 80, evaluation_id="does-not-exist")
        assert exc.value.code == CODE_EVALUATION_NOT_FOUND

    async def test_appends_verdict_without_finalizing_when_below_threshold(self, contract, fake_plugin):
        await register_evaluator(contract, EVALUATOR_1)
        await submit_evaluation(contract, required_verdicts=2)

        result = await submit_verdict(contract, EVALUATOR_1, 80)
        assert not result.HasField("error")

        stored = fake_plugin.store[key_for_evaluation(CONTENT_HASH)]
        evaluation = unmarshal(Evaluation, stored)
        assert evaluation.status == "pending"
        assert len(evaluation.verdicts) == 1

    async def test_auto_finalizes_on_consensus_and_updates_reputation(self, contract, fake_plugin):
        await register_evaluator(contract, EVALUATOR_1, "llama-3.3-70b")
        await register_evaluator(contract, EVALUATOR_2, "mixtral-8x7b")
        await submit_evaluation(contract, required_verdicts=2)

        await submit_verdict(contract, EVALUATOR_1, 80)
        result = await submit_verdict(contract, EVALUATOR_2, 85)  # within tolerance -> consensus
        assert not result.HasField("error")

        evaluation = unmarshal(Evaluation, fake_plugin.store[key_for_evaluation(CONTENT_HASH)])
        assert evaluation.status == "finalized"
        assert evaluation.final_score == round((80 + 85) / 2)

        reputation = unmarshal(Reputation, fake_plugin.store[key_for_reputation(SUBMITTER)])
        assert reputation.score == evaluation.final_score

    async def test_auto_flags_on_disagreement(self, contract, fake_plugin):
        await register_evaluator(contract, EVALUATOR_1, "llama-3.3-70b")
        await register_evaluator(contract, EVALUATOR_2, "mixtral-8x7b")
        await submit_evaluation(contract, required_verdicts=2)

        await submit_verdict(contract, EVALUATOR_1, 20)
        result = await submit_verdict(contract, EVALUATOR_2, 90)  # way outside tolerance -> flagged
        assert not result.HasField("error")

        evaluation = unmarshal(Evaluation, fake_plugin.store[key_for_evaluation(CONTENT_HASH)])
        assert evaluation.status == "flagged"
        assert key_for_reputation(SUBMITTER) not in fake_plugin.store

    async def test_rejects_verdict_on_already_finalized_evaluation(self, contract):
        await register_evaluator(contract, EVALUATOR_1)
        await register_evaluator(contract, EVALUATOR_2)
        await register_evaluator(contract, EVALUATOR_3)
        await submit_evaluation(contract, required_verdicts=2)

        await submit_verdict(contract, EVALUATOR_1, 80)
        await submit_verdict(contract, EVALUATOR_2, 82)  # finalizes here

        with pytest.raises(PluginError) as exc:
            await submit_verdict(contract, EVALUATOR_3, 81)
        assert exc.value.code == CODE_EVALUATION_NOT_PENDING
