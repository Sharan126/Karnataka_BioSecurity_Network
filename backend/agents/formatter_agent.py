import re

class FormatterAgent:
    """
    Agent 4: Formatter Agent
    Converts raw outputs from Image, Retriever, and Reasoning agents into a standardized JSON response.
    """
    def execute(self, incident_obj, retrieved_context, reasoning_text):
        # Parse reasoning sections
        concern = self._extract_section(reasoning_text, "POSSIBLE CONCERN")
        precautions = self._extract_section(reasoning_text, "IMMEDIATE PRECAUTIONS")
        isolation = self._extract_section(reasoning_text, "ISOLATION RECOMMENDATION")
        medicines = self._extract_section(reasoning_text, "RECOMMENDED MEDICINES & THERAPEUTICS") or self._extract_section(reasoning_text, "RECOMMENDED MEDICINES")
        farmer_adv = self._extract_section(reasoning_text, "FARMER ADVISORY")
        vet_adv = self._extract_section(reasoning_text, "VETERINARY ADVISORY")
        govt_rec = self._extract_section(reasoning_text, "GOVERNMENT REPORTING RECOMMENDATION")

        # Format farmer recommended steps
        farmer_recommendations = []
        for line in farmer_adv.split("\n"):
            line_clean = re.sub(r"^[\d\-\*\.]+\s*", "", line.strip())
            if line_clean:
                farmer_recommendations.append(line_clean)
        
        if not farmer_recommendations:
            farmer_recommendations = [
                "Separate the animal from the herd immediately",
                "Avoid herd contact and restrict footwear movement",
                "Contact nearby veterinarian for official examination"
            ]

        # Format medicines list
        meds_list = []
        if medicines:
            for line in medicines.split("\n"):
                line_clean = re.sub(r"^[\d\-\*\.]+\s*", "", line.strip())
                if line_clean and not line_clean.lower().startswith("insufficient evidence"):
                    meds_list.append(line_clean)

        if not meds_list:
            symptoms_str = str(incident_obj.get("symptoms_observed") or incident_obj.get("symptoms") or "").lower()
            anim = incident_obj.get("animal_type", "cattle").title()
            if any(k in symptoms_str for k in ['hair loss', 'skin', 'crust', 'lesion', 'red', 'scab', 'itch', 'alopecia']):
                meds_list = [
                    f"Antiparasitic / Acaricide: Subcutaneous Ivermectin @ 0.2 mg/kg or topical Permethrin spray for {anim}.",
                    "Topical Antiseptic Wash: Clean skin lesions twice daily with 1% Chlorhexidine or Povidone-Iodine solution.",
                    "Anti-inflammatory / Pain Control: NSAID (Meloxicam 0.5 mg/kg) under veterinary supervision.",
                    "Supportive Skin Therapy: Vitamin A, D3, E and Zinc oral supplement to promote skin recovery."
                ]
            elif any(k in symptoms_str for k in ['cough', 'nasal', 'breath', 'pneumonia', 'fever', 'respiratory']):
                meds_list = [
                    "Broad-Spectrum Antibiotic: Enrofloxacin @ 5 mg/kg or Oxytetracycline @ 20 mg/kg IM for secondary pneumonia.",
                    "Antipyretic / NSAID: Meloxicam or Flunixin Meglumine for fever and pulmonary inflammation.",
                    "Supportive Hydration: Oral Rehydration Salts (ORS), dextrose, and warm clean water."
                ]
            else:
                meds_list = [
                    f"Parenteral Antimicrobial: Broad-spectrum antibiotic coverage as prescribed by licensed veterinarian for {anim}.",
                    "Antipyretic / Analgesic: NSAID (Meloxicam 0.5 mg/kg) for temperature regulation and pain relief.",
                    "Supportive Therapy: Oral Rehydration Salts (ORS) and high-energy nutritional supplement."
                ]

        # Determine district outbreak trigger
        severity = incident_obj.get("severity", "medium").lower()
        create_outbreak = severity in ["medium", "high", "critical"]
        
        risk_level = "green"
        if severity in ["high", "critical"]:
            risk_level = "red"
        elif severity == "medium":
            risk_level = "yellow"

        formatted_output = {
            "gemma_output": {
                "animal": incident_obj.get("animal_type", "cattle"),
                "visible_abnormalities": incident_obj.get("symptoms_observed", "Physical symptoms observed"),
                "possible_concern": incident_obj.get("issue_title", "Livestock distress"),
                "urgency": severity,
                "confidence": incident_obj.get("confidence", 0.88),
                "requires_vet_review": incident_obj.get("needs_vet_visit", True),
                "farmer_action": incident_obj.get("description", "Isolate animal and seek veterinary examination.")
            },
            "incident": {
                "animal_type": incident_obj.get("animal_type"),
                "issue_title": incident_obj.get("issue_title"),
                "symptoms": incident_obj.get("symptoms_observed"),
                "description": incident_obj.get("description"),
                "severity": severity,
                "confidence": incident_obj.get("confidence", 0.88),
                "needs_vet_visit": incident_obj.get("needs_vet_visit", True)
            },
            "retrieved_context": retrieved_context,
            "farmer_response": {
                "observed": incident_obj.get("symptoms_observed"),
                "recommended": farmer_recommendations[:4],
                "recommended_medicines": meds_list,
                "disclaimer": "This is not a confirmed diagnosis. A licensed veterinarian must examine the animal and prescribe exact medication."
            },
            "vet_summary": {
                "possible_concern": concern if concern else "Requires clinical examination",
                "immediate_precautions": precautions if precautions else "Isolate and disinfect premises",
                "isolation_recommendation": isolation if isolation else "Isolate animal at least 100 meters from herd",
                "recommended_medicines": meds_list,
                "vet_advisory": vet_adv if vet_adv else "Clinical evaluation and diagnostic sample collection advised",
                "requires_vet_visit": incident_obj.get("needs_vet_visit", True)
            },
            "district_alert": {
                "create_outbreak": create_outbreak,
                "risk_level": risk_level,
                "recommendation": govt_rec if govt_rec else "Monitor 3 km surveillance zone and enforce farm biosecurity"
            }
        }

        return formatted_output

    def _extract_section(self, text, header_name):
        pattern = rf"{header_name}:\s*(.*?)(?=\n[A-Z\s]+:|\Z)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""
