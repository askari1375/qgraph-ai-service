from src.api.schemas.search import RequesterContext, SearchPlanRequest
from src.services.planning import POLICY_LABEL, build_planning_response, choose_planning_mode


def test_choose_planning_mode_is_sync_for_guest():
    requester = RequesterContext(is_authenticated=False, is_guest=True)
    assert choose_planning_mode(requester) == "sync"


def test_choose_planning_mode_is_sync_for_authenticated_user():
    requester = RequesterContext(is_authenticated=True, is_guest=False)
    assert choose_planning_mode(requester) == "sync"


def test_build_planning_response_uses_real_policy_label():
    response = build_planning_response(SearchPlanRequest(query="mercy"))
    assert response.mode == "sync"
    assert response.policy_label == POLICY_LABEL
    assert response.policy_label != "mock_v1"


def test_build_planning_response_defaults_to_guest_without_context():
    response = build_planning_response(SearchPlanRequest(query="mercy"))
    assert response.policy_snapshot["requester"] == {"is_authenticated": False, "is_guest": True}


def test_build_planning_response_echoes_requester_context():
    request = SearchPlanRequest(
        query="mercy",
        context={"requester": {"is_authenticated": True, "is_guest": False}},
    )
    response = build_planning_response(request)
    assert response.policy_snapshot["requester"] == {"is_authenticated": True, "is_guest": False}
