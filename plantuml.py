print("--- Generating Component Diagram ---")
component_puml_raw = get_completion(component_diagram_prompt, client, model_name, api_provider)
component_puml = clean_llm_output(component_puml_raw, language='plantuml')

print("\n--- Generated PlantUML Code ---")
print(component_puml)

component_uml_path = "artifacts/component_diagram.puml"
save_artifact(component_puml, component_uml_path)

# Render the diagram
if component_puml:
    render_plantuml_diagram(component_uml_path, "artifacts/day2_sp_chat_app_component_diagram.png")