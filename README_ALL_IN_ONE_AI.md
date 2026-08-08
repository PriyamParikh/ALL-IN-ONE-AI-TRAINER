# All in One AI — Production 5-in-1 STEM Trainer

**All in One AI** combines five independent desktop AI laboratories in one color-coded launcher.

## Five AI studios

| No. | Studio | Color | Main purpose |
|---|---|---|---|
| 01 | Image AI | Blue | Classify a complete image or webcam frame |
| 02 | Object AI | Green | Train and test a YOLO multi-object detector |
| 03 | Pose AI | Purple | Train named human poses using YOLO keypoints |
| 04 | Audio AI | Orange | Train microphone-based sound classes |
| 05 | Data AI | Teal | Train classification or regression from CSV/Excel |

The four already-tested trainers are included unchanged. The new **Data AI** module adds supervised machine learning for tabular datasets.

## Installation

1. Extract the complete ZIP to a short path, for example:

```text
C:\AllInOneAI
```

2. Double-click:

```text
Install_All_in_One_AI.bat
```

3. Double-click:

```text
Verify_All_in_One_AI.bat
```

4. Start the suite:

```text
Run_All_in_One_AI.bat
```

## Data AI workflow

1. Upload a `.csv`, `.tsv`, `.xlsx` or `.xls` file.
2. The GUI profiles every loaded column: name, detected semantic type, pandas dtype, missing values, unique values and an example.
3. Select the target column—the answer the AI should predict.
4. Use Auto Detect, Classification or Regression.
5. Review automatically suggested ID/constant exclusions.
6. Set independent train, validation and test percentages.
7. Select exactly four or five models.
8. Train and compare models.
9. Generate research graphs.
10. Save the full `.joblib` pipeline or batch-predict a new data file.

### Classification models

- Random Forest
- Decision Tree
- SVM
- KNN
- Logistic Regression
- Neural Network

### Regression models

- Random Forest
- Decision Tree
- SVR
- KNN Regressor
- Ridge Regression
- Neural Network

### Classification graphs

- Testing confusion matrix
- Validation confusion matrix
- ROC curve
- Precision–Recall curve
- Model comparison
- Target distribution
- Missing-values graph
- Numeric correlation heatmap
- PCA/SVD feature scatter
- Permutation feature importance
- Learning curve

### Regression graphs

- Testing and validation actual-versus-predicted graphs
- Residual plot
- Residual distribution
- Model comparison
- Target distribution
- Missing-values graph
- Numeric correlation heatmap
- PCA/SVD feature scatter
- Permutation feature importance
- Learning curve

## Large-file behavior

CSV files are scanned in chunks. The default retained-row cap is 50,000, selected through random-priority sampling across the full CSV. This permits inspection and training on very large files without attempting to place every row in RAM. Set the cap to `0` only when the computer has enough memory for all rows.

Excel files are loaded through pandas. For large workbooks, set an appropriate row cap.

## Model formats

| Studio | Model format |
|---|---|
| Image AI | `.joblib` |
| Object AI | `.pt` |
| Pose AI | `.joblib` plus pretrained YOLO pose weights |
| Audio AI | `.joblib` |
| Data AI | `.joblib` complete preprocessing and estimator pipeline |

## Important classroom guidance

- Use small, distinct classes for the first activity.
- Record or collect balanced examples.
- Keep separate validation and testing data.
- Do not treat a high test score as proof that a model will work in every real-world condition.
- Review confusion matrices, residuals and feature-importance graphs rather than accuracy alone.
