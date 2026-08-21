from search_state import PaperSearchState


def test_state_initialization():
    state = PaperSearchState(
        question="What is the effect of X?"
    )

    assert state.question == "What is the effect of X?"
    assert state.action_history == ()
    assert state.reasoning_history == ()
    assert state.sub_questions == ()
    assert state.retrieved_knowledge == ()
    assert state.summaries == ()
    assert state.candidate_answer is None


def test_child_state_preserves_question():
    root = PaperSearchState(
        question="What is the effect of X?"
    )

    child = root.child(
        action="A4"
    )

    assert child.question == root.question


def test_reasoning_history_propagates():
    root = PaperSearchState(
        question="What is the effect of X?"
    )

    child = root.child(
        action="A4",
        reasoning="Search for evidence about X.",
    )

    grandchild = child.child(
        action="A3",
        reasoning="Investigate the relationship between X and Y.",
    )

    assert grandchild.reasoning_history == (
        "Search for evidence about X.",
        "Investigate the relationship between X and Y.",
    )


def test_evidence_propagates():
    root = PaperSearchState(
        question="What is the effect of X?"
    )

    a4 = root.child(
        action="A4",
        retrieved_knowledge=["Evidence A"],
    )

    a3 = a4.child(
        action="A3",
        retrieved_knowledge=["Evidence B"],
    )

    assert a3.retrieved_knowledge == (
        "Evidence A",
        "Evidence B",
    )


def test_subquestions_propagate():
    root = PaperSearchState(
        question="What is the effect of X?"
    )

    child = root.child(
        action="A4",
        sub_questions=["What is X?"],
    )

    grandchild = child.child(
        action="A3",
        sub_questions=["How does X affect Y?"],
    )

    assert grandchild.sub_questions == (
        "What is X?",
        "How does X affect Y?",
    )


def test_summaries_propagate():
    root = PaperSearchState(
        question="What is the effect of X?"
    )

    child = root.child(
        action="A4",
        summary="Paper A discusses X.",
    )

    grandchild = child.child(
        action="A3",
        summary="Paper B connects X and Y.",
    )

    assert grandchild.summaries == (
        "Paper A discusses X.",
        "Paper B connects X and Y.",
    )


def test_action_history_propagates():
    root = PaperSearchState(
        question="What is the effect of X?"
    )

    child = root.child(
        action="A4"
    )

    grandchild = child.child(
        action="A3"
    )

    assert grandchild.action_history == (
        "A4",
        "A3",
    )


def test_parent_is_not_modified():
    root = PaperSearchState(
        question="What is the effect of X?"
    )

    child = root.child(
        action="A4",
        reasoning="Reasoning A",
        retrieved_knowledge=["Evidence A"],
    )

    assert root.action_history == ()
    assert root.reasoning_history == ()
    assert root.retrieved_knowledge == ()

    assert child.action_history == ("A4",)
    assert child.reasoning_history == ("Reasoning A",)
    assert child.retrieved_knowledge == ("Evidence A",)


def test_branches_are_independent():
    root = PaperSearchState(
        question="What is the effect of X?"
    )

    branch_a = root.child(
        action="A4",
        reasoning="Follow branch A.",
        retrieved_knowledge=["Evidence A"],
    )

    branch_b = root.child(
        action="A5",
        reasoning="Follow branch B.",
        retrieved_knowledge=["Evidence B"],
    )

    assert branch_a.action_history == ("A4",)
    assert branch_b.action_history == ("A5",)

    assert branch_a.reasoning_history == (
        "Follow branch A.",
    )

    assert branch_b.reasoning_history == (
        "Follow branch B.",
    )

    assert branch_a.retrieved_knowledge == (
        "Evidence A",
    )

    assert branch_b.retrieved_knowledge == (
        "Evidence B",
    )


def test_deep_trajectory_is_cumulative():
    root = PaperSearchState(
        question="What is the effect of X?"
    )

    a4 = root.child(
        action="A4",
        reasoning="Find evidence.",
        retrieved_knowledge=["Evidence A"],
    )

    a3 = a4.child(
        action="A3",
        reasoning="Analyze evidence.",
        sub_questions=["Why does X matter?"],
    )

    a5 = a3.child(
        action="A5",
        reasoning="Search for supporting evidence.",
        retrieved_knowledge=["Evidence B"],
        summary="Combined evidence.",
    )

    assert a5.action_history == (
        "A4",
        "A3",
        "A5",
    )

    assert a5.reasoning_history == (
        "Find evidence.",
        "Analyze evidence.",
        "Search for supporting evidence.",
    )

    assert a5.sub_questions == (
        "Why does X matter?",
    )

    assert a5.retrieved_knowledge == (
        "Evidence A",
        "Evidence B",
    )

    assert a5.summaries == (
        "Combined evidence.",
    )


def test_state_is_immutable():
    root = PaperSearchState(
        question="What is the effect of X?"
    )

    try:
        root.question = "Changed question"
        assert False, "State should be immutable"
    except AttributeError:
        pass