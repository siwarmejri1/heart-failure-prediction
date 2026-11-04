# Prédiction de maladie cardiaque (Heart Failure Prediction)

Exploration, préparation des données et modèle de classification, avec explication des prédictions.

## Données
[Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) (Kaggle) : 918 patients, 11 variables cliniques, cible `HeartDisease` (508 cas positifs, 410 négatifs). Voir `data/README.md`.

## Méthode
1. Exploration : types, statistiques, distribution de la cible, variables catégorielles.
2. Nettoyage : les valeurs à 0 de `Cholesterol` et `RestingBP` sont physiologiquement impossibles, donc traitées comme manquantes et remplacées par la **médiane**.
3. Encodage des variables catégorielles et normalisation (MinMaxScaler).
4. **Sélection de variables** par deux méthodes : test univarié (SelectKBest, ANOVA) et RFE avec Random Forest.
5. Modèle **Random Forest** avec séparation train/test, évalué par précision, rappel et F1.
6. Explicabilité avec **SHAP** (importance globale des variables).

## Résultats
| Classe | Précision | Rappel | F1 |
|---|---|---|---|
| 0 (sain) | 0,87 | 0,89 | 0,88 |
| 1 (maladie) | 0,91 | 0,89 | 0,90 |

Exactitude globale de **0,89** sur 184 patients de test. Les variables les plus informatives sont `ST_Slope`, `ExerciseAngina` et `Oldpeak`.

## Limites et pistes d'amélioration
- La normalisation et la sélection de variables sont faites sur l'ensemble des données avant la séparation train/test : il vaut mieux les placer dans un `Pipeline` scikit-learn pour éviter toute fuite d'information.
- Une seule séparation train/test et pas de réglage des hyperparamètres : ajouter une validation croisée et comparer d'autres modèles (régression logistique, gradient boosting).
- En contexte médical, le rappel de la classe « maladie » est plus important que l'exactitude.

## Lancer le projet
```bash
pip install -r requirements.txt
jupyter notebook heart_failure_prediction.ipynb
```

**Stack :** Python, Pandas, scikit-learn, SHAP, Matplotlib, Seaborn
