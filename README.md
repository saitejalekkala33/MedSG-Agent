
# MedSG Agent — Tools + Agent wiring

This repo organizes Task 1,2,3,5,6,7 code, exposes them as **LangChain tools**:
- `examples_all_tasks.py` — one file with **all examples**, ready to toggle by commenting/uncommenting.
- `medsg_agent/tools.py` — LangChain `StructuredTool`s for each task
- `medsg_agent/agent.py` — an `initialize_agent` wiring for OpenAI models
- Common helpers consolidated in `medsg_agent/utils.py`

## Install
```bash
pip install -r requirements.txt
```

## File tree
```
medsg_agent/
  __init__.py
  agent.py
  tools.py
  utils.py
  tasks/
    __init__.py
    task1_registered_diff.py
    task2_nonregistered_diff.py
    task3_multi_view.py
    task5_concept_match.py
    task6_patch_grounding.py
    task7_crossmodal.py
examples_all_tasks.py
requirements.txt
```

## Usage — Tools in an Agent
```python
from medsg_agent import build_agent

agent = build_agent(model_name="gpt-4o-mini", temperature=0.1)
# Example high-level instruction:
q = (
    "Given these two images, use preprocessing to align (if needed), find differences, and output bbox JSON. "
    '{"image_a":"path/to/A.png","image_b":"path/to/B.png"}'
)
print(agent.run(q))
```

## Usage — Run Examples (all tasks in one file)
Open `examples_all_tasks.py` and uncomment the functions you want, then:
```bash
python examples_all_tasks.py
```

### Notes
- All reusable functions are in `utils.py` to avoid duplication.
- Tools return a JSON string: `{"bbox": {...} | null, "confidence": float, "details": {...}}`.
- Images are expected on disk; pass paths in tool args.
- For multi-view/cross-modal tasks, provide the reference bbox where applicable.
