# Jak funguje ML v aplikaci - jednoduché vysvětlení

## ✅ ZÁVĚR: Oba používají STEJNÝ ML model!

### Backtest vs AI Recommendations

**BACKTEST:**
1. Stáhne 2+ let historických dat
2. Pro každý den v minulosti:
   - Natrénuje ML model na datech do tohoto dne
   - Použije model pro predikci následujícího dne
   - Porovná predikci se skutečnou cenou
3. Vypočítá metriky: R², MAE, Direction Accuracy, atd.
4. **Účel:** Otestovat, jak dobře by model fungoval v minulosti

**AI RECOMMENDATIONS:**
1. Stáhne 2+ let historických dat
2. Natrénuje ML model JEDNOU na všech datech
3. Použije model pro predikci BUDOUCÍ ceny (1m, 3m, 6m, 12m)
4. Model se uloží do cache (aby nemusel trénovat pokaždé)
5. **Účel:** Předpovědět budoucí cenu pro investiční doporučení

### Co mají SPOLEČNÉ:

✅ **Stejná funkce:** `_train_random_forest_model()`
✅ **Stejná data:** 2+ let historických dat
✅ **Stejné features:** Technické indikátory, fundamentální data
✅ **Stejný algoritmus:** Random Forest s TimeSeriesSplit cross-validation

### Jak poznat, jestli se používá ML nebo fallback?

V kódu:
- `result['model_used'] == 'random_forest'` → ✅ SKUTEČNÝ ML
- `result['model_used'] == 'momentum_estimate'` → ⚠️ FALLBACK (není ML)

V AI Recommendations:
- Pokud model trénování uspěje → použije se ML
- Pokud selže → použije se momentum fallback s warning

### Test ukázal:

```
✅ POUŽÍVÁ SKUTEČNÝ ML MODEL!
- Model použit: random_forest
- Training R² score: 0.9871 (velmi dobrý!)
- Natrénoval se na 440 samples s 38 features
```

**Takže ANO - AI Recommendations používají stejný ML model jako Backtest!**

