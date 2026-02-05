# WeatherGuardTN : Tunisia Weather Danger & Vigilance Predictor

**Protecting lives, students, delivery workers, fishermen, vulnerable people — and helping authorities prepare in advance**

## Project Title

**WeatherGuardTN**  


## Objectives

The main goal of this application is to **save lives, reduce material damage, and help society prepare better** by providing **earlier, more localized, and more personalized** predictions of dangerous weather than waiting only for official announcements.

### Target users & concrete use-cases

| User group                        | Real-life need / decision to support                                                                 | Type of alert / output we want to provide                                      |
|-----------------------------------|-------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| Students & parents                | Know if classes will be suspended ("arrêt des cours") or exams postponed the next day                 | Probability of school closure / "Plan B likely"                                 |
| Delivery drivers                  | Decide whether to accept dangerous deliveries (rain, flood, strong wind, extreme heat)               | "Avoid delivery" recommendation + risk level                                    |
| Fishermen & mariners              | Know if going out to sea is life-threatening (high waves, strong winds, storms)                      | "Do not go out" / "Return to port" probability                                  |
| General population                | Receive early personal vigilance advice: stay home, avoid travel, protect property                   | Color-coded risk levels similar to official vigilance map (green → red)         |
| **Governments / Civil Protection / Authorities** | Be ready **before** the situation becomes critical: prepare infrastructure, alert teams, open shelters, clean drains, position pumps, warn hospitals, etc. | Early probability of **orange/red vigilance level** per governorate + expected intensity (rain volume, wind gusts, flood risk zones) |
| NGOs & local associations         | Anticipate needs for food/water distribution, evacuation support, elderly checks during heatwaves/floods | Aggregated risk dashboard per region + historical comparison                    |

### Types of dangers predicted

The system forecasts the probability of **dangerous conditions** including:

- Extreme high temperature (heatwave – risk of heatstroke)
- Heavy rain & flash floods
- Very strong winds & storms
- Risk of school closure ("arrêt des cours")
- Dangerous sea conditions (for fishermen)
- Combined multi-hazard risk (rain + wind + heat + potential infrastructure overload)

### Core features (current & planned)

- Multi-source data collection (historical + live scraping + open APIs)
- Machine learning prediction of risk levels (XGBoost / time-series models)
- Per-governorate forecasts (Tunis, Ariana, Sousse, Jendouba, Sfax, Bizerte…)
- Web dashboard (React) with:
  - Region selector
  - Risk color cards (green / yellow / orange / red)
  - Probability % + concrete recommendations ("Stay home", "No delivery", "No classes likely", "Do not go to sea", "Prepare pumps & teams")
- Automated daily data refresh & model inference
- Early warning indicators that could help Civil Protection anticipate orange/red alerts by several hours

### Why this project matters in Tunisia

- Tunisia regularly experiences deadly flash floods (January 2026 events, previous years in Nabeul, Bizerte, Siliana…)
- Heatwaves are becoming more frequent and dangerous
- School closures happen several times per rainy season — parents & students need earlier visibility
- Delivery workers (Talabt, Jumia, Glovo…) face real danger during bad weather
- Small fishermen lose lives almost every year during winter storms
- **Civil Protection and local authorities often have only a few hours** between the INM vigilance bulletin and real impact — early ML-based indicators could give them precious extra preparation time (cleaning drains, positioning teams, alerting hospitals, opening shelters…)

→ **Early, local, personalized prediction can save lives, reduce economic losses, and make emergency response more effective.**

## Tech Stack (high level)

- Backend: FastAPI (Python) + XGBoost / Prophet
- Frontend: React (Vite)
- Automation: n8n workflows (scheduled scraping & prediction)
- Data: Open-Meteo archive, Visual Crossing, scraped BBC / timeanddate + INM vigilance
- Deployment target: Docker Compose → Render / Railway / VPS

## Current Status (February 2026)

- Core scraping & data merging logic working
- Basic XGBoost classifier trained on precipitation + wind → school closure proxy
- React dashboard prototype (region selector + mock risk cards)
- Next steps: real-time vigilance color prediction, sea-state risk, multi-hazard scoring, authority-oriented early-warning dashboard

---

Made with ❤️ in Tunis  
For Tunisians — by a Tunisian

Any feedback, additional data sources, or collaboration welcome!
