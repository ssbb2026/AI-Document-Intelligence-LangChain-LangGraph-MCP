import re

def chunk_document(
    text: str,
    source: str,
    chunk_size: int = 1200,
    overlap: int = 200
):

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0"
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative"
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    # --------------------------------------------------------
    # Clean text
    # --------------------------------------------------------

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    text = text.strip()

    if not text:
        raise ValueError(
            "No text was extracted from the PDF."
        )

    # --------------------------------------------------------
    # Markdown headings
    # --------------------------------------------------------

    lines = text.split("\n")

    current_chapter = None
    current_section = None
    current_subsection = None

    sections = []
    current_content = []

    # --------------------------------------------------------
    # Save section
    # --------------------------------------------------------

    def save_section():

        if current_content:

            content = "\n".join(
                current_content
            ).strip()

            if content:

                sections.append({
                    "chapter": current_chapter,
                    "section": current_section,
                    "subsection": current_subsection,
                    "text": content
                })

    # --------------------------------------------------------
    # Build hierarchical sections
    # --------------------------------------------------------

    for line in lines:

        line = line.strip()

        if not line:

            if (
                current_content
                and current_content[-1] != ""
            ):
                current_content.append("")

            continue

        # H1
        if re.match(r"^#\s+", line):

            save_section()

            current_content = []

            current_chapter = re.sub(
                r"^#\s+",
                "",
                line
            )

            current_section = None
            current_subsection = None

            current_content.append(line)

        # H2
        elif re.match(r"^##\s+", line):

            save_section()

            current_content = []

            current_section = re.sub(
                r"^##\s+",
                "",
                line
            )

            current_subsection = None

            current_content.append(line)

        # H3
        elif re.match(r"^###\s+", line):

            save_section()

            current_content = []

            current_subsection = re.sub(
                r"^###\s+",
                "",
                line
            )

            current_content.append(line)

        else:

            current_content.append(line)

    save_section()

    # ========================================================
    # Split sections into chunks
    # ========================================================

    documents = []

    chunk_counter = 0

    for section in sections:

        section_text = section["text"]

        # ----------------------------------------------------
        # Small section
        # ----------------------------------------------------

        if len(section_text) <= chunk_size:

            documents.append({

                "id": f"chunk_{chunk_counter}",

                "source": source,

                "chunk_index": chunk_counter,

                "chapter": section["chapter"],

                "section": section["section"],

                "subsection": section["subsection"],

                "text": section_text

            })

            chunk_counter += 1

            continue

        # ----------------------------------------------------
        # Large section
        # ----------------------------------------------------

        paragraphs = re.split(
            r"\n\s*\n",
            section_text
        )

        current_chunk = ""

        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if not paragraph:
                continue

            # ------------------------------------------------
            # Paragraph fits
            # ------------------------------------------------

            if (
                len(current_chunk)
                + len(paragraph)
                + 2
                <= chunk_size
            ):

                if current_chunk:
                    current_chunk += "\n\n"

                current_chunk += paragraph

            # ------------------------------------------------
            # Paragraph does not fit
            # ------------------------------------------------

            else:

                if current_chunk:

                    documents.append({

                        "id": f"chunk_{chunk_counter}",

                        "source": source,

                        "chunk_index": chunk_counter,

                        "chapter": section["chapter"],

                        "section": section["section"],

                        "subsection": section["subsection"],

                        "text": current_chunk.strip()

                    })

                    chunk_counter += 1

                # ------------------------------------------------
                # Very large paragraph
                # ------------------------------------------------

                if len(paragraph) > chunk_size:

                    start = 0

                    while start < len(paragraph):

                        end = start + chunk_size

                        small_chunk = (
                            paragraph[start:end]
                            .strip()
                        )

                        if small_chunk:

                            documents.append({

                                "id": (
                                    f"chunk_{chunk_counter}"
                                ),

                                "source": source,

                                "chunk_index":
                                    chunk_counter,

                                "chapter":
                                    section["chapter"],

                                "section":
                                    section["section"],

                                "subsection":
                                    section["subsection"],

                                "text":
                                    small_chunk

                            })

                            chunk_counter += 1

                        start = end - overlap

                    current_chunk = ""

                else:

                    current_chunk = paragraph

        # ----------------------------------------------------
        # Save final chunk
        # ----------------------------------------------------

        if current_chunk:

            documents.append({

                "id": f"chunk_{chunk_counter}",

                "source": source,

                "chunk_index": chunk_counter,

                "chapter": section["chapter"],

                "section": section["section"],

                "subsection": section["subsection"],

                "text": current_chunk.strip()

            })

            chunk_counter += 1

    # ========================================================
    # Previous / next relationships
    # ========================================================

    for i, document in enumerate(documents):

        document["previous_chunk"] = (
            documents[i - 1]["id"]
            if i > 0
            else None
        )

        document["next_chunk"] = (
            documents[i + 1]["id"]
            if i < len(documents) - 1
            else None
        )

    return documents
