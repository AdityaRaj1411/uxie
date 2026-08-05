import pytest
from src.rbac.permissions import Permission, evaluate_permission, ActionType

def test_evaluate_permission():
    assert evaluate_permission(ActionType.SEARCH) == Permission.READER
    assert evaluate_permission(ActionType.SAVE_NEW) == Permission.WRITER
    assert evaluate_permission(ActionType.MODIFY_EXISTING) == Permission.EDITOR
    assert evaluate_permission(ActionType.DELETE) == Permission.ADMIN
