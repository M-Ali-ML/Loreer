# Loreer (WIB)
Your friendly wise LLM that knows all things [League of Legends (LoL)](https://www.leagueoflegends.com/en-us/) lore.
A RAG system that runs locally using Llama.cpp and extract relevant information from LoL [Wiki website](https://leagueoflegends.fandom.com/wiki/League_of_Legends_Wiki) stored as a [Vector Embeddings](https://en.wikipedia.org/wiki/Vector_database).

Below is a high-level overview of the sytem.
![Loreer-design (1)](./assets/Loreer-design.png)
###### [link to whiteboard](https://whimsical.com/loreer-design-DQbf9fQs6HfMCJRimPsLon)

## Repo Structure
There are 4 main python notebooks which are:
| Name | Objective | Path |
|----------|----------|----------|
| 00-parse_xml_dumb.ipynb | parses the XML file and removes unnecessary pages | [Link](./00-parse_xml_dumb.ipynb) |
| 01-preprocess_data.ipynb | clean the data and split it into chunks | [Link](./01-preprocess_data.ipynb) |
| 02-embed_chunks.ipynb | Embedd the chunks into a vector embedding for fast retrival | [Link](./02-embed_chunks.ipynb) |
| 03-pure_LLM.ipynb | running a quantized Llama-3-8b locally using llama.cpp | [Link](./03-pure_LLM.ipynb) |
| 04-RAG_system.ipynb | Integerate the embedded query and the LLM into a single prompt | [Link](./04-RAG_system.ipynb) |

## Reproducibility 
The repository's code serves as a strong foundation for the potential extension of this RAG system to encompass different Wiki/fandom domains.
