from pathlib import Path


STATIC_DIR = Path(__file__).parents[1] / "factlayer" / "static"


def test_ui_assets_exist_without_a_build_step() -> None:
    assert (STATIC_DIR / "index.html").is_file()
    assert (STATIC_DIR / "styles.css").is_file()
    assert (STATIC_DIR / "app.js").is_file()


def test_ui_exposes_required_workflows() -> None:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    assert 'id="upload-form"' in html
    assert 'type="file"' in html
    assert 'data-view="facts"' in html
    assert 'data-view="relations"' in html
    assert 'data-view="failures"' in html
    assert 'id="detail-drawer"' in html
    assert 'aria-live="polite"' in html


def test_ui_calls_the_planned_evidence_first_api() -> None:
    javascript = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    for endpoint in (
        'request("/health")',
        'request("/documents")',
        'request("/failures?limit=200")',
        'request(`/facts/${id}`)',
        'request(`/facts/${factId}/evidence`)',
    ):
        assert endpoint in javascript

    # Relations are fetched by type from the server rather than filtered in the browser.
    # Filtering one fixed page client-side made the interesting verdicts unreachable: the
    # first page came back entirely needs_review, so corroborates, contradicts and
    # reconciled all rendered as empty while thousands of them sat in the database.
    assert "relation_type=" in javascript
    assert "loadRelations" in javascript
    # "reconciled" is a family of stored types, so the filter has to fan out to all of them.
    for member in ("reconciled_by_scope", "reconciled_by_as_of", "reconciled_by_period"):
        assert member in javascript
    # Per-document fact pages are merged round-robin so one document cannot fill the list.
    assert "interleave" in javascript

    assert "FormData" in javascript
    assert "escapeHtml" in javascript
    assert "API unavailable" in javascript
