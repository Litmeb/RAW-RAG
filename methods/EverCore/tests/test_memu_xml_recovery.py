from evaluation.src.adapters.memu_xml_recovery import (
    parse_memory_type_response_xml,
)


def test_valid_xml_preserves_entities_and_escapes_bare_ampersands():
    raw = """
    Some prose before the XML.
    <knowledge>
      <memory>
        <content>R&amp;D uses A & B</content>
        <categories><category>work &amp; research</category></categories>
      </memory>
    </knowledge>
    Some prose after the XML.
    """

    assert parse_memory_type_response_xml(raw) == [
        {
            "content": "R&D uses A & B",
            "categories": ["work & research"],
        }
    ]


def test_mismatched_tag_recovers_memory_instead_of_dropping_response():
    raw = """
    ```xml
    <events>
      <memory>
        <content>Alice visited Paris</content>
        <categories>
          <category>travel</category>
          <category>Paris</content>
        </categories>
      </memory>
    </events>
    ```
    """

    assert parse_memory_type_response_xml(raw) == [
        {
            "content": "Alice visited Paris",
            "categories": ["travel", "Paris"],
        }
    ]


def test_one_bad_memory_does_not_discard_other_memory_blocks():
    raw = """
    <profile>
      <memory>
        <content>Alice likes tea</content>
        <categories><category>preference</category></categories>
      </memory>
      <memory>
        <content>Incomplete record</content>
      </memory>
      <memory>
        <content>Alice lives in Shanghai</wrong>
        <categories><category>location</category></categories>
      </memory>
    </profile>
    """

    assert parse_memory_type_response_xml(raw) == [
        {"content": "Alice likes tea", "categories": ["preference"]},
        {"content": "Alice lives in Shanghai</wrong>", "categories": ["location"]},
    ]


def test_missing_supported_root_returns_empty_list():
    assert parse_memory_type_response_xml("<memory>no root</memory>") == []
