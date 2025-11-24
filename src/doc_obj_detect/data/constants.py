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
