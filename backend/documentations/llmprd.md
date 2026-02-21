PRD: LLM Modules — Preventive Risk-Based Stock Replenishment System
System Context (Non-LLM Modules — For Reference Only)
The LLM modules described in this PRD exist within a larger four-module pipeline. The coding agent does not implement the following components, but should be aware of their interfaces as they produce inputs consumed by the LLM layers.
Module 1A — Topological Epidemiological Risk Analyzer (Non-LLM)
Applies Topological Data Analysis (TDA) to regional health and geospatial data, computing persistent homology to track structural changes in disease case distributions over time. An Isolation Forest model flags anomalous shifts in persistence diagrams. This module outputs a JSON payload of per-region anomaly scores and topological signals that are passed directly into Module 1B (an LLM module in scope).
Module 3 — Logistics Routing Engine (Non-LLM)
A deterministic formula-based engine that computes optimal drug delivery routes across pharmacy locations. It factors in geographic distance, delivery vehicle capacity, urgency tiers, and regional demand weights. Outputs a ranked routing plan per pharmacy that is consumed by the Final Orchestration Agent.
These modules will be developed independently and integrated at a later stage. The coding agent should stub their outputs using the mock schemas defined in this PRD to enable isolated development and testing of the LLM pipeline.

Scope
Three LLM deliverables:

Module 1B — Regional Disease Spread Risk Narrator
Module 2 — Pharmacy Inventory Gap Analyzer
Final Orchestration Agent — Multi-input replenishment decision agent


Module 1B: Regional Disease Spread Risk Narrator
Purpose
Consume anomaly signals from the upstream TDA + Isolation Forest pipeline and generate a structured risk assessment per region, including disease focus recommendations that downstream modules use to prioritise stock decisions.
Input
json{
  "regions": [
    {
      "region_id": "string",
      "region_name": "string",
      "anomaly_score": "float (0–1)",
      "persistence_shift": "float",
      "flagged": "boolean",
      "historical_outbreaks": ["string"],
      "population_density": "string",
      "season": "string"
    }
  ]
}
Output
json{
  "risk_assessments": [
    {
      "region_id": "string",
      "risk_level": "LOW | MEDIUM | HIGH | CRITICAL",
      "spread_likelihood": "float (0–1)",
      "reasoning": "string",
      "recommended_disease_focus": ["string"]
    }
  ]
}
Behaviour

The LLM must reason over anomaly scores, persistence shifts, seasonal context, and historical outbreak data to produce a calibrated likelihood score.
Risk levels must follow a defined threshold mapping enforced in the prompt — not left to model discretion. Suggested mapping: 0.0–0.25 = LOW, 0.26–0.50 = MEDIUM, 0.51–0.75 = HIGH, 0.76–1.0 = CRITICAL.
The reasoning field must be 2–4 sentences, factual in tone, and explicitly reference the input signals that drove the assessment.
Unflagged regions must still receive a LOW risk assessment — they must not be omitted from the output.
Recommended disease focus should name specific disease categories (e.g. "respiratory infections", "dengue fever") rather than generic terms.

Implementation Notes

Use structured output parsing to enforce JSON schema on every call.
System prompt must frame the model as an epidemiological risk analyst.
Temperature: 0.2.


Module 2: Pharmacy Inventory Gap Analyzer
Purpose
Scan a hardcoded pharmacy inventory dataset — including supplier and pricing information — against disease risk signals from Module 1B. Identify stock gaps, shortfalls, near-expiry stock, and cost-significant misallocations that require intervention.
Input
Hardcoded Inventory Dataset (injected as static context, not a dynamic API call):
json{
  "pharmacies": [
    {
      "pharmacy_id": "string",
      "pharmacy_name": "string",
      "location": "string",
      "region_id": "string",
      "stock": [
        {
          "drug_name": "string",
          "category": "string",
          "quantity": "integer",
          "unit_price_usd": "float",
          "reorder_threshold": "integer",
          "reorder_quantity": "integer",
          "days_until_expiry": "integer",
          "supplier_id": "string",
          "supplier_name": "string",
          "supplier_lead_time_days": "integer",
          "supplier_unit_cost_usd": "float"
        }
      ]
    }
  ]
}
Risk Context (passed from Module 1B):
json{
  "risk_assessments": [
    {
      "region_id": "string",
      "risk_level": "string",
      "recommended_disease_focus": ["string"]
    }
  ]
}
Output
json{
  "gap_reports": [
    {
      "pharmacy_id": "string",
      "pharmacy_name": "string",
      "location": "string",
      "critical_gaps": [
        {
          "drug_name": "string",
          "category": "string",
          "issue": "UNDERSTOCKED | MISSING | NEAR_EXPIRY",
          "current_quantity": "integer",
          "recommended_quantity": "integer",
          "urgency": "LOW | MEDIUM | HIGH",
          "estimated_restock_cost_usd": "float",
          "preferred_supplier_id": "string",
          "preferred_supplier_name": "string",
          "supplier_lead_time_days": "integer"
        }
      ],
      "total_estimated_restock_cost_usd": "float",
      "overall_readiness_score": "float (0–1)",
      "summary": "string"
    }
  ]
}
Behaviour

Drugs relevant to disease categories flagged by Module 1B for a pharmacy's region must be prioritised in the gap analysis and assigned higher urgency regardless of current stock levels.
Stock below the reorder threshold must be flagged as UNDERSTOCKED with a recommended quantity equal to the reorder quantity defined in the inventory data.
Drugs completely absent from a pharmacy's stock but relevant to the risk context must be flagged as MISSING. The LLM should recommend a reorder quantity proportional to the region's risk level and population density context passed from Module 1B.
Near-expiry stock with fewer than 30 days remaining must always be flagged as NEAR_EXPIRY regardless of quantity or risk context.
Estimated restock cost must be computed as recommended_quantity × supplier_unit_cost_usd and included per gap item. The total across all gap items must be summed into total_estimated_restock_cost_usd.
When multiple suppliers are available in the dataset for the same drug, the LLM should prefer the supplier with the lowest unit cost unless lead time exceeds urgency window, in which case it should prefer the fastest supplier.
The readiness score must reflect how well the pharmacy is equipped for the predicted regional disease burden, not just raw stock levels.
Summary must be 1–3 sentences, written in actionable operational language.

Implementation Notes

Inventory dataset is hardcoded and injected into the system prompt or as a document block. It does not come from an API call.
Estimated cost calculations should be validated post-generation by a Python function rather than trusted blindly from the LLM output.
Use structured output parsing with Pydantic. Temperature: 0.1.


Module 3 (Non-LLM Stub): Logistics Routing Engine
The coding agent must implement a mock stub of Module 3 that returns the following fixed schema so the Final Orchestration Agent can be developed and tested independently. The real routing engine will replace this stub at integration time.
json{
  "routing_plan": [
    {
      "pharmacy_id": "string",
      "location": "string",
      "delivery_urgency_score": "float (0–1)",
      "estimated_delivery_time_hours": "integer",
      "capacity_remaining_units": "integer",
      "available_suppliers": [
        {
          "supplier_id": "string",
          "supplier_name": "string",
          "distance_km": "float",
          "can_fulfil_by_urgency": "boolean"
        }
      ]
    }
  ]
}

Final Orchestration Agent
Purpose
Act as the central decision-making agent. Ingest outputs from all three upstream modules and produce a final replenishment directive specifying which drugs to send, in what quantities, from which supplier, at what cost, and to which pharmacy — ranked by priority.
Input
json{
  "risk_assessments": [ "...Module 1B output..." ],
  "gap_reports": [ "...Module 2 output..." ],
  "routing_plan": [ "...Module 3 output..." ]
}
Output
json{
  "replenishment_directives": [
    {
      "pharmacy_id": "string",
      "pharmacy_name": "string",
      "location": "string",
      "priority_rank": "integer",
      "assigned_supplier_id": "string",
      "assigned_supplier_name": "string",
      "drugs_to_deliver": [
        {
          "drug_name": "string",
          "category": "string",
          "quantity": "integer",
          "unit_cost_usd": "float",
          "total_cost_usd": "float",
          "rationale": "string"
        }
      ],
      "total_order_cost_usd": "float",
      "delivery_window": "string",
      "feasibility_flag": "FEASIBLE | CONSTRAINED | INFEASIBLE",
      "directive_summary": "string"
    }
  ],
  "grand_total_cost_usd": "float",
  "overall_system_summary": "string"
}
Behaviour

The agent must reconcile three potentially conflicting signals — regional risk severity, inventory gaps, and logistical feasibility — and must not blindly prioritise any single input.
Priority rank must be assigned globally across all pharmacies as a single ordered list. Rank 1 is the most urgent delivery.
Drug quantities in the directive must be grounded in the gap reports from Module 2. The agent must not fabricate quantities without basis in the gap data.
Supplier assignment must be consistent with the available suppliers listed in the routing plan for that pharmacy. The agent should prefer suppliers flagged as can_fulfil_by_urgency: true for HIGH and CRITICAL risk regions.
Unit and total costs must be carried through from Module 2 gap report data. A post-generation Python validator must verify that quantity × unit_cost_usd = total_cost_usd and that line items sum correctly to total_order_cost_usd and then to grand_total_cost_usd.
The rationale per drug must explicitly name which input signal drove the decision: risk level from Module 1B, gap type from Module 2, or routing constraint from Module 3.
If a high-priority delivery is logistically constrained or infeasible, the feasibility_flag must reflect this and the directive_summary must explicitly describe the constraint rather than silently deprioritising the pharmacy.
The overall system summary must be 3–5 sentences written for a non-technical operations manager. It should cover total cost, highest priority deliveries, and any feasibility concerns.

Implementation Notes

Implement as a LangGraph node downstream of all three module nodes.
Use tool calling or structured output to enforce the response schema.
System prompt must frame the agent as a pharmaceutical supply chain operations coordinator.
Temperature: 0.2.
Include a schema validation step post-generation. If the schema is violated or cost arithmetic fails, re-invoke the LLM with the specific errors appended as feedback before raising an exception. Maximum one retry.


Shared Implementation Requirements
LLM Provider: OpenAI GPT-4o or Anthropic Claude Sonnet, configurable via environment variable LLM_PROVIDER. API keys must be loaded from environment variables, never hardcoded.
Output Validation: All three modules must define Pydantic v2 models for their response schemas. Cost arithmetic fields must additionally be verified by a Python post-processing function. Schema violations trigger one automatic retry with error context appended before raising a structured exception.
Prompt Management: All system prompts and user prompt templates must be stored as .jinja2 files in a /prompts directory. No prompts are to be hardcoded inline in application logic.
Logging: Every LLM call must log the following to stdout in structured JSON: module name, timestamp, input token count, output token count, latency in milliseconds, retry triggered (boolean), and any validation errors encountered.
Error Handling: If an LLM call fails after retry, the module must return a structured error object of the form { "module": "string", "error": "string", "input_snapshot": {} } rather than raising an unhandled exception, enabling the orchestration agent to gracefully degrade.
Mock Data: The coding agent must provide a realistic hardcoded dataset for Module 2 covering at least 5 pharmacies across 3 regions, with at least 10 drugs per pharmacy including varied stock levels, expiry dates, supplier IDs, and pricing. This dataset should be stored as a JSON file in /data/inventory.json.
Testing: Each module must have a standalone test script that runs the module against mock inputs and prints the structured output. These are not unit tests — they are integration smoke tests to verify LLM output schema compliance end to end.
Environment: Python 3.11+, LangGraph for orchestration, LangChain for LLM abstraction, Pydantic v2 for schema validation.

Out of Scope for Coding Agent

TDA computation and Isolation Forest model (Module 1A)
Real routing formula engine (Module 3)
Frontend, dashboard, or API layer
Authentication, rate limiting, or deployment infrastructure
Real pharmacy, patient, or supplier data — all datasets are hardcoded or mocked
Database integration — all state is in-memory for this development phase


