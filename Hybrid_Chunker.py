#!/usr/bin/env python3
import os
import re
import json
import argparse # Imported for command-line argument parsing
from typing import List, Dict, Any
from pathlib import Path

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Consider using a more accurate tokenizer, e.g., from tiktoken,
# especially if aligning with specific LLM tokenization.
# import tiktoken

class LangGraphChunker:
    HEADER_PATTERN = re.compile(r'^(#{1,4})\s+(.*)$', re.MULTILINE)

    def __init__(
        self,
        base_chunk_size: int = 1200,
        chunk_overlap: int = 200,
        code_chunk_buffer: int = 300,
        min_full_code_tokens: int = 50,
        code_ratio_threshold: float = 0.3
    ):
        self.base_chunk_size = base_chunk_size
        self.chunk_overlap = chunk_overlap
        self.code_chunk_buffer = code_chunk_buffer
        self.min_full_code_tokens = min_full_code_tokens
        self.code_ratio_threshold = code_ratio_threshold
        self.chunk_counter = 0

        # Concepts for tagging, can be expanded or refined
        self.langgraph_concepts = {
            "quickstart": ["quickstart", "introduction", "get started", "setup", "installation"],
            "state": ["state", "stategraph", "state management", "add_node", "add_edge"],
            "api_reference": ["api", "class", "method", "function", "parameter", "reference"],
            "concept": ["concept", "overview", "architecture"],
            "workflow": ["workflow", "graph", "nodes", "edges"],
            "chatbot": ["chatbot", "conversational agent"],
        }

    def _count_tokens(self, text: str) -> int:
        # This is an approximate token count. For more accuracy, consider using a
        # model-specific tokenizer (e.g., tiktoken for OpenAI models, or a
        # tokenizer aligned with Google's embedding models).
        # Example with tiktoken (requires installation and model specification):
        # enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        # return len(enc.encode(text))
        return len(text.split())

    def _split_by_headers(self, text: str) -> List[Document]:
        # Splits the document based on markdown headers (H1-H4).
        # Each resulting document carries metadata of the headers it's under.
        docs = []
        current_lines = []
        current_meta = {"h1": "", "h2": "", "h3": "", "h4": ""}
        for line in text.splitlines(keepends=True):
            m = self.HEADER_PATTERN.match(line)
            if m:
                if current_lines: # Save previous section before starting a new one
                    docs.append(Document("".join(current_lines), metadata=current_meta.copy()))
                    current_lines = []
                level = len(m.group(1))
                current_meta[f"h{level}"] = m.group(2).strip()
                # Reset deeper level headers
                for l_reset in range(level + 1, 5):
                    current_meta[f"h{l_reset}"] = ""
            current_lines.append(line)
        if current_lines: # Append the last section
            docs.append(Document("".join(current_lines), metadata=current_meta.copy()))
        return docs

    def _extract_code_blocks(self, text: str) -> List[Dict[str, Any]]:
        # Extracts all code blocks (```...```) from the text.
        # Returns a list of dicts, each with start/end position, code, and token count.
        blocks = []
        for m in re.finditer(r"```[\s\S]*?```", text, re.DOTALL):
            code = m.group(0)
            tokens = self._count_tokens(code)
            blocks.append({"start": m.start(), "end": m.end(), "code": code, "tokens": tokens})
        return blocks

    def _dynamic_chunk_size(self, text: str) -> int:
        # Adjusts chunk size based on the text's token count.
        # Aims for smaller chunks for shorter texts and caps at a max relative to base_chunk_size.
        tokens = self._count_tokens(text)
        if tokens < self.base_chunk_size // 2:
            size = tokens
        elif tokens > 2 * self.base_chunk_size:
            # Ensure size is not excessively large, but also not too small
            size = max(self.base_chunk_size // 2, self.base_chunk_size)
        else:
            size = self.base_chunk_size
        # Ensure chunk size is at least overlap + 1 to avoid issues with splitter
        return max(size, self.chunk_overlap + 1)

    def _semantic_split(self, text: str, size: int) -> List[str]:
        # Splits text into chunks of a given 'size' using RecursiveCharacterTextSplitter.
        # Tries to split along semantic boundaries (paragraphs, sentences).
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""], # Common separators
        )
        return splitter.split_text(text.strip())

    def _extract_tags(self, text: str, headers: Dict[str,str]) -> List[str]:
        # Extracts predefined tags based on keywords found in text and headers.
        # Helps in categorizing chunks.
        combined = (text + " " + " ".join(headers.values())).lower()
        tags = ["langgraph"] # Default tag
        for concept, kws in self.langgraph_concepts.items():
            if any(kw in combined for kw in kws):
                tags.append(concept)
        return list(dict.fromkeys(tags)) # Remove duplicates while preserving order

    def _determine_doc_type(self, text: str, headers: Dict[str,str]) -> str:
        # Heuristically determines the type of the document/chunk.
        # Accuracy here is important for later filtering in retrieval.
        lower = (text + " " + " ".join(headers.values())).lower()
        if "```" in text: # Presence of code block often indicates an example
            return "code_example"
        if "api" in lower or "parameter" in lower:
            return "api_reference"
        if "concept" in lower or "overview" in lower: # Keywords for conceptual docs
            return "concept"
        return "documentation" # Default type

    def _make_chunk(
        self,
        content: str,
        section: str, # Top-level section (e.g., filename or main doc area)
        subsection: str, # More granular subsection (e.g., specific part of a file)
        headers: Dict[str,str], # Markdown headers associated with this chunk
        includes_code: bool, # Does the chunk contain any code block?
        is_full: bool, # Is this chunk primarily a self-contained code example?
        part_example: bool # Is this chunk part of a larger code example context?
    ) -> Dict[str,Any]:
        # Constructs the chunk dictionary with content and metadata.
        self.chunk_counter += 1
        return {
            "chunk_id": self.chunk_counter,
            "section": section,
            "subsection": subsection,
            "content": content.strip(),
            "metadata": {
                "includes_code": includes_code,
                "is_full_code_example": is_full,
                "part_of_code_example": part_example,
                "num_code_blocks": content.count("```"),
                "headers": headers, # Store H1-H4 headers
                "tags": self._extract_tags(content, headers),
                "topic": self._determine_doc_type(content, headers)
            }
        }

    def process(self, text: str, section: str, subsection: str) -> List[Dict[str,Any]]:
        # Main processing logic for a given text.
        # 1. Splits by headers.
        # 2. For each header-defined section:
        #    a. Extracts code blocks.
        #    b. Differentiates processing for code-heavy vs. prose-heavy content.
        #    c. Code-heavy: Isolates large code blocks, then chunks surrounding prose.
        #    d. Prose-heavy: Isolates large code blocks, then chunks remaining prose with dynamic sizing.
        # This logic is complex; thorough testing with varied inputs is recommended.
        # Adding logging here can help debug chunking behavior.
        docs = self._split_by_headers(text)
        all_chunks = []

        for doc in docs:
            content = doc.page_content
            headers = {k:v for k,v in doc.metadata.items() if v} # Filter out empty headers

            code_blocks = self._extract_code_blocks(content)
            total_tokens = self._count_tokens(content)
            code_tokens = sum(b["tokens"] for b in code_blocks)
            code_ratio = code_tokens / total_tokens if total_tokens else 0.0

            # Strategy 1: Code-heavy content
            # If a significant portion of the content is code, prioritize extracting full code blocks.
            if code_ratio >= self.code_ratio_threshold:
                for blk in code_blocks:
                    if blk["tokens"] >= self.min_full_code_tokens:
                        # Attempt to include some preceding context for the code block
                        pre_context_text = content[:blk["start"]]
                        # Take last N words as context (tunable)
                        context_words = pre_context_text.split()[-self.code_chunk_buffer:]
                        context = " ".join(context_words)
                        
                        all_chunks.append(self._make_chunk(
                            f"{context}\n\n{blk['code']}", # Prepend context
                            section, subsection, headers,
                            includes_code=True,
                            is_full=True, # Mark as a full code example
                            part_example=True # Assumed part of a larger example context
                        ))
                # Chunk remaining prose (text not part of extracted code blocks)
                prose_content = content
                for blk in code_blocks: # Remove already processed code blocks
                    prose_content = prose_content.replace(blk["code"], "")
                # Use a smaller chunk size for surrounding prose in code-heavy sections
                prose_chunk_size = max(self.base_chunk_size // 2, 100)
                for part in self._semantic_split(prose_content, prose_chunk_size):
                    all_chunks.append(self._make_chunk(
                        part, section, subsection, headers,
                        includes_code="```" in part,
                        is_full=False,
                        part_example="```" in part # If it still contains small code snippets
                    ))

            # Strategy 2: Prose-heavy content
            else:
                # First, extract any large, self-contained code examples
                for blk in code_blocks:
                    # Condition: code block is large enough AND it's near the end of the doc
                    # (implying it might be a concluding example)
                    if blk["tokens"] >= self.min_full_code_tokens and \
                       (total_tokens - blk["end"] < self.chunk_overlap * 2):
                        pre_context_text = content[:blk["start"]]
                        context_words = pre_context_text.split()[-self.code_chunk_buffer:]
                        context = " ".join(context_words)
                        all_chunks.append(self._make_chunk(
                            f"{context}\n\n{blk['code']}",
                            section, subsection, headers,
                            includes_code=True,
                            is_full=True,
                            part_example=True
                        ))
                
                # Chunk remaining prose after removing explicitly extracted full code examples
                prose_content = content
                for blk in code_blocks:
                    if blk["tokens"] >= self.min_full_code_tokens: # Only remove large ones
                        prose_content = prose_content.replace(blk["code"], "")
                
                # Use dynamic chunk sizing for the main prose content
                dynamic_size = self._dynamic_chunk_size(prose_content)
                for part in self._semantic_split(prose_content, dynamic_size):
                    all_chunks.append(self._make_chunk(
                        part, section, subsection, headers,
                        includes_code="```" in part,
                        is_full=False, # Not a full code example chunk by default
                        part_example="```" in part
                    ))
        return all_chunks

    def save(self, chunks: List[Dict[str,Any]], output_path: str):
        # Saves the list of chunks to a JSON file.
        # Ensures the output directory exists.
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Error saving chunks to {output_path}: {e}")
            # Potentially re-raise or handle more gracefully

    def run_file(self, input_filepath: str, output_filepath: str, section_name: str):
        # Processes a single input file and saves its chunks.
        try:
            text = Path(input_filepath).read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"Error: Input file not found at {input_filepath}")
            return
        except Exception as e:
            print(f"Error reading file {input_filepath}: {e}")
            return
            
        subsection_name = Path(input_filepath).stem # Use filename (without ext) as subsection
        
        print(f"Processing {input_filepath} for section '{section_name}', subsection '{subsection_name}'...")
        chunks = self.process(text, section_name, subsection_name)
        self.save(chunks, output_filepath)
        print(f"✅ Created {len(chunks)} chunks from {input_filepath} → {output_filepath}")


def main():
    # Setup argparse for command-line arguments
    parser = argparse.ArgumentParser(description="Chunk document(s) for LangGraph RAG.")
    parser.add_argument(
        "--input",
        type=str,
        default="langgraph.txt", # Default input file 
        help="Path to the input text file or directory of text files."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="D:/SNK/langchain-master/Langraph agent/GLang/langgraph_chunks_refined/chunks_manifest.json", # Default output file
        help="Path to the output JSON manifest file for chunks."
    )
    parser.add_argument(
        "--section",
        type=str,
        default="langgraph", # Default section name
        help="Main section name to assign to these chunks."
    )
    # Add more arguments for chunker parameters if desired
    parser.add_argument("--base_chunk_size", type=int, default=1200)
    parser.add_argument("--chunk_overlap", type=int, default=200)
    parser.add_argument("--code_chunk_buffer", type=int, default=300)
    parser.add_argument("--min_full_code_tokens", type=int, default=50)
    parser.add_argument("--code_ratio_threshold", type=float, default=0.3)

    args = parser.parse_args()

    chunker = LangGraphChunker(
        base_chunk_size=args.base_chunk_size,
        chunk_overlap=args.chunk_overlap,
        code_chunk_buffer=args.code_chunk_buffer,
        min_full_code_tokens=args.min_full_code_tokens,
        code_ratio_threshold=args.code_ratio_threshold
    )

    # Simple handling for a single file input for now.
    # Could be extended to handle a directory of files.
    if Path(args.input).is_file():
        chunker.run_file(args.input, args.output, args.section)
    elif Path(args.input).is_dir():
        print(f"Input is a directory. Processing all .txt files in {args.input}")
        # This part is a placeholder for directory processing logic.
        # You would iterate over files, potentially creating multiple output files
        # or a consolidated manifest. For simplicity, this example will error.
        # For now, we'll just process one file if it's a dir for demo.
        # A more robust solution would be to aggregate chunks from multiple files.
        # Or, save each file's chunks to a separate manifest in the output dir.
        
        # Example: Process first .txt file found, or handle as error
        # This is a simplified approach.
        # A better way would be to collect all chunks and save once,
        # or save individual manifests.
        
        # For now, let's assume if it's a dir, we expect one manifest from all files.
        all_processed_chunks = []
        input_dir = Path(args.input)
        output_dir = Path(os.path.dirname(args.output))
        os.makedirs(output_dir, exist_ok=True)

        for filepath in input_dir.glob("*.txt"): # Or other relevant extensions
            try:
                text = filepath.read_text(encoding="utf-8")
                subsection_name = filepath.stem
                print(f"Processing {filepath} for section '{args.section}', subsection '{subsection_name}'...")
                file_chunks = chunker.process(text, args.section, subsection_name)
                all_processed_chunks.extend(file_chunks)
            except Exception as e:
                print(f"Error processing file {filepath}: {e}")
        
        if all_processed_chunks:
            chunker.save(all_processed_chunks, args.output)
            print(f"✅ Created a total of {len(all_processed_chunks)} chunks from directory {args.input} → {args.output}")
        else:
            print(f"No processable files found or no chunks generated from directory {args.input}")

    else:
        print(f"Error: Input path {args.input} is not a valid file or directory.")

if __name__=="__main__":
    main()
