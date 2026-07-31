import os
import requests

class ReasoningAgent:
    """
    Agent 3: Reasoning Agent
    Synthesizes Gemma/Gemini LLM reasoning, local SOP context, and global veterinary medical knowledge.
    Dynamically generates best global medicines, therapeutics, APIs, dosages, and administration routes
    specifically tailored to the reported animal species and observed symptoms.
    """
    def execute(self, incident_obj, retrieved_context):
        gemma_key = os.environ.get('GEMMA_API_KEY') or os.environ.get('GEMINI_API_KEY')
        
        animal_type = incident_obj.get("animal_type", "cattle")
        symptoms = incident_obj.get("symptoms_observed", incident_obj.get("symptoms", "Physical discomfort"))
        title = incident_obj.get("issue_title", "Livestock distress")
        description = incident_obj.get("description", "")
        severity = incident_obj.get("severity", "medium")

        # Build context string from retrieved chunks
        context_str = ""
        if retrieved_context:
            for idx, c in enumerate(retrieved_context, 1):
                context_str += f"\n--- RETRIEVED DOCUMENT {idx} (Source: {c['source']}, Title: {c['title']}, Confidence: {c['confidence']}) ---\n{c['content']}\n"
        else:
            context_str = "No specific local SOP document found."

        prompt = f"""You are an expert Global Veterinary Clinical Pharmacologist and Chief Medical Officer for the Karnataka BioSecurity Network.

INCIDENT DETAILS REPORTED:
- Animal Species: {animal_type}
- Issue Title: {title}
- Observed Symptoms / Abnormalities: {symptoms}
- Incident Description: {description}
- Urgency / Severity: {severity}

RETRIEVED LOCAL SOP CONTEXT:
{context_str}

CRITICAL INSTRUCTIONS FOR GLOBAL VETERINARY MEDICINES & DIAGNOSIS:
1. DIAGNOSIS: Determine the exact probable diagnosis matching the symptoms ({symptoms}) in {animal_type}. (e.g. Mange / Ringworm / Dermatitis / Lumpy Skin Disease / FMD / PPR / Swine Fever / Avian Influenza / Mastitis / Pneumonia).
2. GLOBAL MEDICINES & THERAPEUTICS: Consult global veterinary pharmacopeia, WOAH/OIE, FAO, and international veterinary standards. Provide the BEST, MOST EFFECTIVE medicines, Active Pharmaceutical Ingredients (APIs), exact dosage guidelines, administration routes (IM, IV, Topical, Oral), and global generic/brand therapeutics tailored specifically to {symptoms} in {animal_type}.
   - If skin lesions/hair loss/crusts: Include antiparasitics (e.g. Ivermectin 0.2 mg/kg), topical antiseptics (Chlorhexidine/Povidone-Iodine), anti-inflammatory NSAIDs (Meloxicam), and Vitamin A/E skin supplements.
   - If blisters/oral lesions: Include oral washes, mouth soothing gel, NSAIDs, and secondary antibiotic cover.
   - If respiratory/cough/fever: Include NSAID antipyretics, broad-spectrum antibiotics (Oxytetracycline/Enrofloxacin), and electrolyte hydration.
   - DO NOT hardcode Foot and Mouth Disease (FMD) mouth wash if the symptoms indicate skin mange, ringworm, or non-FMD conditions!

Please format your response clearly with the following section headers:
POSSIBLE CONCERN: <exact likely diagnosis based on symptoms and global veterinary medical knowledge>
IMMEDIATE PRECAUTIONS: <bullet list of 3-4 immediate biosecurity and farm precautions>
ISOLATION RECOMMENDATION: <specific isolation radius and hygiene protocol>
RECOMMENDED MEDICINES & THERAPEUTICS: <bullet list of specific global medicines, active ingredients, exact dosage, administration routes, and topical washes tailored to this condition>
FARMER ADVISORY: <3 clear actionable steps for the farmer>
VETERINARY ADVISORY: <technical clinical recommendations, diagnostic tests, and treatment protocol for the visiting veterinarian>
GOVERNMENT REPORTING RECOMMENDATION: <surveillance zone and reporting requirements>"""

        reasoning_text = None

        if gemma_key and not gemma_key.startswith('nvapi-'):
            # Valid models on Google AI Gemini/Gemma endpoints
            candidate_models = [
                'gemini-2.5-flash',
                'gemini-2.0-flash',
                'gemini-1.5-flash',
                'gemma-2-27b-it',
                'gemini-1.5-pro'
            ]
            for model_name in candidate_models:
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemma_key}"
                    payload = {
                        "contents": [{"parts": [{"text": prompt}]}]
                    }
                    res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
                    if res.status_code == 200:
                        candidates = res.json().get('candidates', [])
                        if candidates:
                            parts = candidates[0].get('content', {}).get('parts', [])
                            raw_text = ""
                            for p in parts:
                                if 'text' in p and not p.get('thought'):
                                    raw_text += p['text']
                            if not raw_text.strip():
                                for p in parts:
                                    if 'text' in p:
                                        raw_text += p['text']
                            if raw_text.strip():
                                reasoning_text = raw_text.strip()
                                print(f"[SUCCESS] ReasoningAgent generated global veterinary medicines using {model_name}.")
                                break
                    else:
                        print(f"ReasoningAgent API Status {res.status_code} ({model_name}): {res.text[:120]}")
                except Exception as e:
                    print(f"ReasoningAgent Error ({model_name}): {e}")

        # Dynamic symptom-matched fallback if LLM endpoint unreachable
        if not reasoning_text:
            symptoms_lower = symptoms.lower()
            if any(k in symptoms_lower for k in ['hair loss', 'skin', 'crust', 'lesion', 'red', 'scab', 'itch', 'alopecia']):
                meds_fallback = """- Antiparasitic / Acaricide: Subcutaneous Ivermectin @ 0.2 mg/kg body weight or topical Permethrin spray.
- Topical Antiseptic: Wash lesions daily with 1% Chlorhexidine or Povidone-Iodine solution; apply zinc oxide soothing ointment.
- Anti-inflammatory / Pain Relief: Oral/IM Meloxicam @ 0.5 mg/kg body weight under veterinary supervision.
- Supportive Skin Care: Vitamin A, D3, E and Zinc oral supplement to promote epithelial healing."""
                concern_fallback = f"Suspected Parasitic/Fungal Dermatitis (Mange/Ringworm) in {animal_type.title()}"
            elif any(k in symptoms_lower for k in ['blister', 'drool', 'saliva', 'mouth', 'tongue', 'foot', 'hoof']):
                meds_fallback = """- Oral Cavity Wash: Flush mouth lesions twice daily with 1% Potassium Permanganate (KMnO4) solution or 0.5% Alum solution.
- Foot & Interdigital Care: Clean hoof lesions with 1% KMnO4 wash, dry, and apply Loraxene/antiseptic fly-repellent spray.
- Antipyretic / NSAID: Meloxicam @ 0.5 mg/kg IV/IM for fever and severe lameness pain.
- Secondary Bacterial Umbrella: Long-acting Oxytetracycline @ 20 mg/kg IM if prescribed by VAS."""
                concern_fallback = f"Suspected Vesicular Stomatitis / Foot and Mouth Disease in {animal_type.title()}"
            elif any(k in symptoms_lower for k in ['cough', 'nasal', 'breath', 'pneumonia', 'fever', 'respiratory']):
                meds_fallback = """- Broad-Spectrum Antibiotic: Enrofloxacin @ 5 mg/kg or Oxytetracycline @ 20 mg/kg IM for respiratory secondary infection.
- Anti-inflammatory / Antipyretic: Meloxicam or Flunixin Meglumine @ 1.1-2.2 mg/kg for pulmonary inflammation and high fever.
- Mucolytic & Expectorant: Oral Ammonium Chloride & Potassium Iodide solution.
- Hydration: Oral Rehydration Salts (ORS) with dextrose and warm clean water."""
                concern_fallback = f"Suspected Acute Respiratory Disease / Pneumonia in {animal_type.title()}"
            else:
                meds_fallback = """- Broad-Spectrum Antimicrobial: Administer parenteral antibiotic under veterinary prescription.
- Antipyretic & Analgesic: NSAID (Meloxicam 0.5 mg/kg) for pain and temperature regulation.
- Electrolyte & Fluid Support: Oral Rehydration Salts (ORS) and glucose supplementation.
- Topical Care: Apply antiseptic dressing on any physical skin abrasions."""
                concern_fallback = f"Suspected Clinical Distress in {animal_type.title()}"

            reasoning_text = f"""POSSIBLE CONCERN: {concern_fallback}
IMMEDIATE PRECAUTIONS:
- Isolate affected {animal_type} immediately to prevent spread.
- Disinfect premises daily with 1% Sodium Hypochlorite or 2% Phenol.
- Restrict handler movement and enforce footwear footbaths.
ISOLATION RECOMMENDATION: Maintain isolation at least 100 meters away from unexposed livestock for 14 days.
RECOMMENDED MEDICINES & THERAPEUTICS:
{meds_fallback}
FARMER ADVISORY:
1. Separate the sick animal from the herd immediately.
2. Provide clean drinking water, soft green fodder, and electrolyte gruel.
3. Contact nearest veterinary officer for official clinical examination and prescription.
VETERINARY ADVISORY: Perform clinical examination, collect diagnostic samples, and administer targeted anti-inflammatory & antimicrobial therapy.
GOVERNMENT REPORTING RECOMMENDATION: Report to District Deputy Director within 6 hours if mortality exceeds threshold."""

        return reasoning_text
