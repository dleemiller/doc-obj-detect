"""Dataset label metadata."""

PUBLAYNET_CLASSES: dict[int, str] = {
    0: "text",
    1: "title",
    2: "list",
    3: "table",
    4: "figure",
}

PUBLAYNET_ID_MAPPING: dict[int, int] = {
    1: 0,  # text
    2: 1,  # title
    3: 2,  # list
    4: 3,  # table
    5: 4,  # figure
}

DOCLAYNET_CLASSES: dict[int, str] = {
    0: "Caption",
    1: "Footnote",
    2: "Formula",
    3: "List-item",
    4: "Page-footer",
    5: "Page-header",
    6: "Picture",
    7: "Section-header",
    8: "Table",
    9: "Text",
    10: "Title",
}

# DocSynth-300K to PubLayNet mapping (74 M6Doc classes → 5 PubLayNet classes)
# This mapping is based on semantic similarity analysis of M6Doc element types
# Default mapping: unknown classes → text (class 0)
DOCSYNTH_TO_PUBLAYNET_MAPPING: dict[int, int] = {
    # Text elements → text (0)
    0: 0,  # text
    23: 0,  # paragraph/text block
    25: 0,  # text content
    30: 0,  # text element
    # Title/header elements → title (1)
    1: 1,  # title
    34: 1,  # header
    56: 1,  # section header
    # List elements → list (2)
    2: 2,  # list
    9: 2,  # list item
    # Table elements → table (3)
    3: 3,  # table
    48: 3,  # table cell/content
    # Figure/image elements → figure (4)
    4: 4,  # figure
    63: 4,  # image
    # Default fallback: all unmapped classes → text (0)
    # This will be handled programmatically in the loader
}

# Default class for unmapped DocSynth classes
DOCSYNTH_DEFAULT_CLASS = 0  # Map to "text" for unknown elements
