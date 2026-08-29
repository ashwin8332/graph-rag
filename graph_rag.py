"""Graph RAG on a local LLM.  question -> intent+entities -> subgraph -> answer

pip install langchain-ollama langchain-core networkx   |   ollama pull tinyllama
"""

import json, re
import networkx as nx
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama

FACTS = [  # (subject, relation, object) -- swap in your own domain
    ("Ada Lovelace", "born_in", "London"),
    ("Ada Lovelace", "worked_with", "Charles Babbage"),
    ("Ada Lovelace", "known_for", "first computer program"),
    ("Charles Babbage", "invented", "Analytical Engine"),
    ("Analytical Engine", "type_of", "mechanical computer"),
]
G = nx.MultiDiGraph((s, o, {"r": r}) for s, r, o in FACTS)

llm = ChatOllama(model="tinyllama", temperature=0)
prompt = lambda t: ChatPromptTemplate.from_template(t) | llm | StrOutputParser()


def retrieve(entities, hops=2):
    """Facts within `hops` of any mentioned entity, as readable triples."""
    hits = {n for n in G.nodes for e in entities if e.lower() in n.lower()}
    near = set().union(*(nx.ego_graph(G.to_undirected(as_view=True), n, hops) for n in hits)) if hits else set()
    return "\n".join(f"{s} -[{d['r']}]-> {o}" for s, o, d in G.edges(data=True) if {s, o} & near) or "(none)"


def understand(question):
    """Ask the LLM for intent + entities; fall back to raw words if it won't emit JSON."""
    raw = prompt('Question: {question}\nReply with JSON only: {{"intent": "...", "entities": ["..."]}}').invoke(
        {"question": question}
    )
    try:
        got = json.loads(re.search(r"\{.*\}", raw, re.S).group())
        intent, entities = got["intent"], got["entities"]
    except Exception:
        intent, entities = question, re.findall(r"\w+", question)
    return {"question": question, "intent": intent, "facts": retrieve(entities)}


chain = RunnableLambda(understand) | prompt(
    "Use ONLY these facts; say so if they are insufficient.\n\n"
    "Facts:\n{facts}\n\nIntent: {intent}\nQuestion: {question}"
)

if __name__ == "__main__":
    while q := input("\nAsk (blank to quit): ").strip():
        print(chain.invoke(q))
