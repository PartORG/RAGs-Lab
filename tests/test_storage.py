import pytest

from storage import (
    Saved,
    account_stamp,
    add_user,
    check_password,
    delete_user,
    saved_of,
    user_dir,
)


def test_passwords_are_checked_and_can_be_reset():
    add_user("ann", "correct horse")
    assert check_password("ann", "correct horse")
    assert not check_password("ann", "wrong horse")
    assert not check_password("nobody", "correct horse")
    add_user("ann", "battery staple")  # running adduser again resets the password
    assert check_password("ann", "battery staple")
    assert not check_password("ann", "correct horse")


@pytest.mark.parametrize("name", ["../etc", "Ann", "a/b", "", "x" * 33])
def test_names_that_could_escape_the_user_folder_are_refused(name):
    with pytest.raises(ValueError):
        add_user(name, "long enough")
    with pytest.raises(ValueError):
        user_dir(name)


def test_short_passwords_are_refused():
    with pytest.raises(ValueError):
        add_user("ann", "short")


def test_saved_rows_belong_to_one_owner():
    Saved("ann", "naive_rag").save({"a": 1})
    Saved("ann", "naive_rag").save({"a": 2})  # a save replaces
    Saved("bob", "naive_rag").save({"b": 1})
    assert Saved("ann", "naive_rag").load() == {"a": 2}
    assert Saved("ann", "graph_rag").load() is None
    assert saved_of("ann", "naive") == [Saved("ann", "naive_rag")]


def test_delete_user_removes_the_account_its_indexes_and_files_only():
    for name in ("ann", "bob"):
        add_user(name, "long enough")
        Saved(name, "naive_rag").save({})
        (user_dir(name) / "uploads").mkdir(parents=True)
        (user_dir(name) / "uploads" / "a.txt").write_text("x")

    assert delete_user("ann")
    assert not check_password("ann", "long enough")
    assert saved_of("ann", "") == [] and not user_dir("ann").exists()
    assert check_password("bob", "long enough")
    assert saved_of("bob", "") == [Saved("bob", "naive_rag")] and user_dir("bob").exists()
    assert not delete_user("ann")  # already gone
    with pytest.raises(ValueError):
        delete_user("../bob")


def test_account_stamp_changes_on_reset_and_recreation_and_ends_on_delete():
    add_user("ann", "long enough")
    first = account_stamp("ann")
    assert first and account_stamp("ann") == first  # stable while nothing changes
    add_user("ann", "long enough")  # password reset
    second = account_stamp("ann")
    assert second != first
    delete_user("ann")
    assert account_stamp("ann") is None
    add_user("ann", "long enough")  # a new account under the old name
    assert account_stamp("ann") not in (None, first, second)
