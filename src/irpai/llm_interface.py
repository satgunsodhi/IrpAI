"""
IrpAI Stage 5: LLM Interface & Context Packaging

Two approaches for LLM integration:
- Direct Prompt Injection: Serialize structured context into natural language prompt
- Embedding-Based Retrieval: Semantic search for query-relevant UI elements
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from irpai.config import LLMConfig
from irpai.models import StructuredContext

logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Constructs LLM-optimized prompts from structured screenshot context.

    Approach A (Direct Injection): Serializes full context into prompt.
    Approach B (Embedding Retrieval): Retrieves only relevant elements.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._embedder: Optional[ContextEmbedder] = None

    def build_prompt(
        self,
        user_query: str,
        context: StructuredContext,
    ) -> str:
        """
        Build an LLM prompt with screenshot context.

        Args:
            user_query: User's question about the screenshot
            context: Structured context from fusion module

        Returns:
            Complete prompt string for LLM
        """
        if self.config.approach == "embedding_retrieval":
            return self._build_embedding_prompt(user_query, context)
        else:
            return self._build_direct_prompt(user_query, context)

    def _build_direct_prompt(
        self, user_query: str, context: StructuredContext
    ) -> str:
        """
        Build prompt using direct context injection.

        Includes full platform context, layout summary, and extracted text.
        """
        # Build layout summary
        layout_summary = self._format_layout(context)

        # Build spatial relationships summary
        spatial_summary = ""
        if context.spatial_relationships:
            rels = context.spatial_relationships[:10]
            spatial_lines = [
                f"  - {r.subject} is {r.relation} {r.obj}" for r in rels
            ]
            spatial_summary = (
                "\nSPATIAL RELATIONSHIPS:\n" + "\n".join(spatial_lines)
            )

        # Truncate all_text to fit token budget
        max_text = self.config.max_context_tokens * 3  # rough chars-to-tokens
        all_text = context.all_text[:max_text]
        if len(context.all_text) > max_text:
            all_text += "..."

        prompt = f"""You are analyzing a screenshot from a {context.platform} application.

APPLICATION: {context.app_title}

UI LAYOUT:
{layout_summary}
{spatial_summary}

EXTRACTED TEXT:
{all_text}

USER QUESTION:
{user_query}

Please provide a detailed, helpful response based on the screenshot context above. Reference specific UI elements and text when relevant."""

        return prompt

    def _format_layout(self, context: StructuredContext) -> str:
        """Format layout structure into readable summary."""
        lines = []
        max_per_region = self.config.max_elements_per_region

        for region_name in ["navbar", "sidebar", "main_content", "footer", "modal"]:
            elements = getattr(context.layout, region_name)
            if not elements:
                continue

            elem_descriptions = []
            for elem in elements[:max_per_region]:
                desc = elem.element_type.value
                if elem.text:
                    text_preview = elem.text[:50]
                    if len(elem.text) > 50:
                        text_preview += "..."
                    desc += f' ("{text_preview}")'
                if elem.label:
                    desc += f" [label: {elem.label}]"
                if elem.icon_type:
                    desc += f" [icon: {elem.icon_type}]"
                elem_descriptions.append(desc)

            remaining = max(0, len(elements) - max_per_region)
            summary = ", ".join(elem_descriptions)
            if remaining > 0:
                summary += f", ... and {remaining} more"

            lines.append(f"**{region_name.upper()}**: {summary}")

        return "\n".join(lines) if lines else "No layout regions detected"

    def _build_embedding_prompt(
        self, user_query: str, context: StructuredContext
    ) -> str:
        """
        Build prompt using embedding-based retrieval.

        Only includes UI elements most relevant to the user's query.
        """
        if self._embedder is None:
            self._embedder = ContextEmbedder(self.config.embedding_model)

        # Get relevant elements
        relevant = self._embedder.retrieve_relevant(
            query=user_query,
            context=context,
            top_k=self.config.embedding_top_k,
        )

        # Format relevant elements
        relevant_desc = []
        for item in relevant:
            elem = item["element"]
            desc = f"[{item['region'].upper()}] {elem.element_type.value}"
            if elem.text:
                desc += f': "{elem.text[:80]}"'
            desc += f" (relevance: {item['similarity']:.2f})"
            relevant_desc.append(desc)

        prompt = f"""You are analyzing a screenshot from a {context.platform} application.

APPLICATION: {context.app_title}

MOST RELEVANT UI ELEMENTS for your query:
{chr(10).join(relevant_desc)}

FULL TEXT CONTENT:
{context.all_text[:1500]}

USER QUESTION:
{user_query}

Please provide a detailed, helpful response based on the screenshot context above. Reference specific UI elements and text when relevant."""

        return prompt


class ContextEmbedder:
    """
    Creates semantic embeddings for hierarchical retrieval of UI elements.

    Uses SentenceTransformers for lightweight, fast embedding.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _ensure_loaded(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded embedding model: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )

    def embed_context(self, context: StructuredContext) -> dict[str, Any]:
        """
        Create multi-level embeddings for semantic search.

        Levels:
        1. Page-level: Entire screenshot context
        2. Region-level: Each layout region
        3. Element-level: Individual UI elements with text
        """
        self._ensure_loaded()
        import numpy as np

        embeddings: dict[str, Any] = {}

        # Page-level embedding
        if context.all_text:
            embeddings["page"] = self._model.encode(context.all_text[:1000])

        # Region-level embeddings
        embeddings["regions"] = {}
        for region_name in ["navbar", "sidebar", "main_content", "footer", "modal"]:
            elements = getattr(context.layout, region_name)
            region_text = " ".join(e.text or "" for e in elements).strip()
            if region_text:
                embeddings["regions"][region_name] = self._model.encode(region_text)

        # Element-level embeddings
        embeddings["elements"] = []
        for region_name in ["navbar", "sidebar", "main_content", "footer", "modal"]:
            elements = getattr(context.layout, region_name)
            for elem in elements:
                if elem.text:
                    elem_embedding = self._model.encode(elem.text)
                    embeddings["elements"].append(
                        {
                            "embedding": elem_embedding,
                            "element": elem,
                            "region": region_name,
                        }
                    )

        return embeddings

    def retrieve_relevant(
        self,
        query: str,
        context: StructuredContext,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Find UI elements most relevant to user's query.

        Args:
            query: User's question
            context: Structured context
            top_k: Number of elements to return

        Returns:
            List of relevant elements with similarity scores
        """
        self._ensure_loaded()
        import numpy as np

        embeddings = self.embed_context(context)

        if not embeddings.get("elements"):
            return []

        query_embedding = self._model.encode(query)

        # Compute cosine similarities
        results = []
        for elem_data in embeddings["elements"]:
            elem_emb = elem_data["embedding"]
            similarity = float(
                np.dot(query_embedding, elem_emb)
                / (np.linalg.norm(query_embedding) * np.linalg.norm(elem_emb) + 1e-8)
            )
            results.append(
                {
                    "element": elem_data["element"],
                    "region": elem_data["region"],
                    "similarity": similarity,
                }
            )

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]


class LLMInterface:
    """
    Interface to LLM providers for generating responses.

    Supports:
    - OpenAI (GPT-4, GPT-3.5)
    - Local models (via LangChain)
    - Anthropic (Claude)
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.prompt_builder = PromptBuilder(self.config)

    def generate_response(
        self,
        user_query: str,
        context: StructuredContext,
    ) -> tuple[str, str]:
        """
        Generate LLM response with screenshot context.

        Args:
            user_query: User's question about the screenshot
            context: Structured context from fusion module

        Returns:
            Tuple of (enhanced_prompt, llm_response)
        """
        # Build enhanced prompt
        enhanced_prompt = self.prompt_builder.build_prompt(user_query, context)

        # Call LLM
        try:
            response = self._call_llm(enhanced_prompt)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            response = self._generate_fallback_response(user_query, context)

        return enhanced_prompt, response

    def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM provider."""
        import os

        if self.config.llm_provider == "openai":
            return self._call_openai(prompt)
        elif self.config.llm_provider == "anthropic":
            return self._call_anthropic(prompt)
        elif self.config.llm_provider == "local":
            return self._call_local(prompt)
        else:
            raise ValueError(f"Unknown LLM provider: {self.config.llm_provider}")

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        import os

        try:
            from openai import OpenAI

            api_key = os.environ.get(self.config.api_key_env_var)
            if not api_key:
                raise ValueError(
                    f"API key not found. Set {self.config.api_key_env_var} environment variable."
                )

            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=1000,
            )
            return response.choices[0].message.content or ""
        except ImportError:
            raise ImportError("openai not installed. Install with: pip install openai")

    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API."""
        import os

        try:
            import anthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "API key not found. Set ANTHROPIC_API_KEY environment variable."
                )

            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=self.config.llm_model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except ImportError:
            raise ImportError("anthropic not installed. Install with: pip install anthropic")

    def _call_local(self, prompt: str) -> str:
        """Call a local LLM via LangChain."""
        try:
            from langchain_community.llms import Ollama

            llm = Ollama(model=self.config.llm_model)
            return llm.invoke(prompt)
        except ImportError:
            raise ImportError(
                "langchain-community not installed. "
                "Install with: pip install langchain-community"
            )

    def _generate_fallback_response(
        self, user_query: str, context: StructuredContext
    ) -> str:
        """Generate a basic response without LLM when API is unavailable."""
        summary = context.to_summary()
        return (
            f"[IrpAI Offline Mode - LLM unavailable]\n\n"
            f"Screenshot Analysis:\n{summary}\n\n"
            f"Your query: {user_query}\n\n"
            f"Extracted text:\n{context.all_text[:500]}\n\n"
            f"(Connect to an LLM provider for AI-powered responses)"
        )
