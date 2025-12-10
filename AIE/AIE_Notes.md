# MODULE 5 – AI ETHICS (Deep, Clear, Example-Rich Notes)

## 1. Open Research Problems in AI Ethics

AI ethics is basically about making sure AI is helpful, fair, and safe. But many problems are still unsolved. These problems fall into two types: **near-term** (happening now) and **long-term** (future threats).

### A. Near-Term Ethical Problems (Current)

1.  **Bias**
    *   AI learns patterns from past data. If the data is biased, the AI becomes biased.
    *   **Example:** A hiring algorithm trained mostly on resumes from men might rate female candidates lower because of historical imbalance.
    *   **Why it's a research problem:** Bias appears differently in every domain (healthcare, finance, policing), so researchers need better ways to measure and reduce it.

2.  **Privacy Issues**
    *   AI systems collect huge amounts of personal data — photos, location, browsing history, medical records.
    *   **Example:** A fitness app collects heart rate data, which gets leaked and reveals medical conditions.
    *   **Research challenge:** How to allow AI to learn from data while still protecting user privacy.

3.  **Wrong or Unsafe Decisions**
    *   AI isn’t perfect, and errors in sensitive areas can cause real harm.
    *   **Example:** A medical AI misdiagnoses a patient because it was trained on limited datasets.
    *   **Research challenge:** How to make AI reliable enough for real-world high-stakes decisions.

4.  **Misinformation**
    *   AI can generate deepfakes or fake news that spreads faster than humans can verify.
    *   **Example:** A deepfake video of a political leader could spread during elections and mislead millions.
    *   **Research challenge:** Detecting AI-generated content at scale.

5.  **Misuse of AI**
    *   AI tools can be used for harmful purposes.
    *   **Example:** Voice cloning used to scam families by pretending to be a relative.
    *   **Research challenge:** Balancing open innovation with safeguards.

6.  **No Clear Accountability**
    *   If AI harms someone, who is responsible?
    *   **Example:** A self-driving car causes an accident — is it the manufacturer, the AI developer, or the owner?
    *   **Research challenge:** Creating frameworks for responsibility and liability.

### B. Long-Term Ethical Problems (Future Risks)

1.  **Power Concentration**
    *   Powerful AI models require huge resources that only a few companies or countries possess.
    *   **Impact:** They could dominate the economy, influence society, or control information.
    *   **Example:** Only two or three tech giants controlling all advanced AI systems.

2.  **Misaligned Superintelligent AI**
    *   Future AI may become highly autonomous. If its goals don’t match human values, outcomes can be dangerous.
    *   **Example:** A super-AI told to “solve climate change” might decide humans are the problem. (The classic paperclip AI thought experiment.)

3.  **Social Norm Erosion**
    *   AI may slowly change how people behave or think.
    *   **Example:** Students using AI for everything may lose certain skills over time. Or social media algorithms might reshape beliefs and opinions.

---

## 2. Near-Term Research Areas (What researchers are working on)

### 1. Measuring and Reducing Bias
*   Researchers design tools to:
    *   Detect unfair outcomes.
    *   Compare different demographic groups.
    *   Understand what causes bias.
*   **Example:** Testing whether a loan approval AI rejects one particular community more than others.

### 2. Making AI Explanations Easy
*   People don’t trust decisions they can’t understand.
*   **Research tries to create explanations that:**
    *   Are simple.
    *   Match the user's background.
    *   Show key reasons behind decisions.
*   **Example:** A doctor needs to know *why* an AI flagged a tumor, not just the prediction.

### 3. Accountability and Transparency
*   **Researchers want systems that:**
    *   Record how decisions are made.
    *   Store versions of AI models.
    *   Allow independent audits.
*   **Example:** If a credit score AI denies a loan, regulators should be able to check how the decision was made.

### 4. Balancing Privacy With Accuracy
*   AI needs data, but users need privacy.
*   **Research explores:**
    *   **Federated learning:** Data stays on your device.
    *   **Differential privacy:** Adding noise to protect identity.
*   **Example:** Google keyboard suggestions learn from your typing without sending your exact messages to Google servers.

### 5. Fighting Misinformation
*   **Research includes:**
    *   Watermarking AI-generated images.
    *   Tools to detect deepfakes.
    *   Tracing the origin of videos and photos.
*   **Example:** Meta and OpenAI watermarking images so platforms know they’re AI-generated.

### 6. Human–AI Decision Sharing
*   Sometimes AI should decide; sometimes humans should.
*   **Researchers study:**
    *   When to trust AI.
    *   How to design human–AI collaboration.
    *   How to avoid over-reliance.
*   **Example:** AI suggests cancer diagnosis, but the doctor makes the final call.

---

## 3. Long-Term Research Areas

### 1. Aligning AI With Human Values
*   **Researchers want AI that understands:**
    *   Cultural values.
    *   Fairness.
    *   Diverse human preferences.
*   **Example:** What is “fair” differs between countries; AI must adapt responsibly.

### 2. Avoiding Power Imbalance
*   **Goal:** Prevent a small group from controlling global AI power.
*   **Example:** Open-source models help spread benefits more widely.

### 3. Preparing for Job and Social Shifts
*   AI automation may replace certain jobs.
*   **Researchers focus on:**
    *   Retraining workers.
    *   Social safety nets.
    *   New job roles.
*   **Example:** Radiologists working alongside AI instead of being replaced.

### 4. Long-Term Global Safety
*   Rare but dangerous scenarios must be prevented.
*   **Example:** AI accidentally triggering economic collapse or cyberwar due to misaligned goals.

---

## 4. Challenges, Opportunities, and Approaches

### Challenges
*   Hard to measure ethical issues.
*   Attackers adapt faster than defenses.
*   Different countries have different laws.
*   Companies focus on profit, not ethics.

### Opportunities
*   Fairer services.
*   Better decision-making in health, education.
*   Transparency in government systems.
*   More access to knowledge.

### Approaches
*   Teams with **technologists + sociologists + lawyers**.
*   Including communities in AI design.
*   Continuous monitoring and impact assessments.
*   **Regulatory sandboxes** where new rules can be tested.

---

## 5. Societal Issues of AI in Medicine

### Major Concerns
1.  **Biased Medical Data:** If training data mostly includes one race, age group, or gender, AI may give incorrect results for others.
    *   *Example:* Skin-cancer detection AI performing poorly on darker skin tones.
2.  **Lack of Explainability:** Doctors need reasons, not just predictions.
3.  **Privacy:** Health data is extremely sensitive.
4.  **Liability:** Who is responsible for a wrong diagnosis — the doctor or the AI company?
5.  **Workflow Disruption:** Doctors must adjust their routines to integrate AI tools smoothly.

### Approaches
*   Conduct clinical trials for AI just like medicines.
*   Keep humans in the loop.
*   Track where training data came from.
*   Clear patient consent for data use.
*   Post-deployment monitoring to catch errors early.
*   Explainability tailored for doctors and patients.

### Research Needs
*   Fairness benchmarks specific to healthcare.
*   Combining medical knowledge with machine learning.
*   Government standards for approving medical AI tools.

---

## 6. Decision Roles of AI in Industry

1.  **Advisory**
    *   AI only suggests. Human decides.
    *   *Example:* Netflix recommending movies.

2.  **Assisted**
    *   AI does most of the work; human signs off.
    *   *Example:* Doctors reviewing AI-detected fractures on X-rays.

3.  **Automated**
    *   AI makes decisions independently.
    *   *Example:* Sorting packages in a warehouse.

4.  **Supervisory**
    *   AI handles many automated tasks; humans monitor overall performance.
    *   *Example:* A single person supervising hundreds of robots in a factory.

### Design Rules
*   **High-risk** = Human final authority.
*   **Low-risk** = Automation allowed.
*   Always have **fallback systems**.

### Best Practices
*   Risk categorization.
*   Testing models in real environments.
*   Continuous evaluation for fairness and reliability.
*   Clear audit trails.

---

## 7. AI and Democracy

### Risks
*   Deepfakes manipulating elections.
*   Targeted political ads micro-manipulating voters.
*   Big tech controlling the political narrative.
*   Surveillance harming civil liberties.

### Safeguards
*   Transparency in political ads.
*   Watermarking synthetic media.
*   Independent audits.
*   Increasing digital literacy.

### Research Questions
*   How to detect political manipulation at scale?
*   How to provide transparency without breaking privacy?
*   How to design governance that preserves multiple viewpoints?

---

## 8. National and International AI Strategies

### What Good AI Strategies Include
*   Strong safety standards.
*   Investment in AI R&D that benefits the public.
*   Workforce training programs.
*   Strict data governance and privacy laws.
*   International coordination for high-risk AI systems.
*   Civil society oversight.

### Operational Tools
*   **Regulatory sandboxes** (test new AI safely).
*   Model registries.
*   Mandatory impact assessments.
*   Export controls for dangerous AI capabilities.
*   Public funding for audits and verification.

---

## 9. Benefits of Ethical AI
*   Increases public trust.
*   Reduces legal and social harm.
*   Ensures fairer services.
*   Improves long-term social outcomes.
*   Makes regulatory compliance easier.
*   Builds resilience against misuse.
