# Results
Both the sentiment ana
## Dictionary based classifer
### Count Plot
![countPlot](https://github.com/Amloner/Sentiment-Analysis-of-User-Reviews/assets/124287518/5d120772-657d-418a-9c76-5b540ef7c0b9)
### Box Plot
![boxPlot](https://github.com/Amloner/Sentiment-Analysis-of-User-Reviews/assets/124287518/82574744-dab0-4165-a36a-a50022276fc4)
### Confusion Matrix
![confusionMatrix](https://github.com/Amloner/Sentiment-Analysis-of-User-Reviews/assets/124287518/24dbe0cb-d1d3-42a5-8430-104b027843dc)
### Report
|                  | **precision**       | **recall**          | **f1-score**        | **support**        |
| ---------------- | ------------------- | ------------------- | ------------------- | ------------------ |
| **negative**     | 0.5989787902592300  | 0.7625              | 0.6709194896612410  | 2000.0             |
| **neutral**      | 0.09403437815975730 | 0.186               | 0.12491605104096700 | 500.0              |
| **positive**     | 0.7233160621761660  | 0.349               | 0.47082630691399700 | 2000.0             |
| **accuracy**     | 0.5146666666666670  | 0.5146666666666670  | 0.5146666666666670  | 0.5146666666666670 |
| **macro avg**    | 0.4721097435317180  | 0.43250000000000000 | 0.4222206158720680  | 4500.0             |
| **weighted avg** | 0.5981348653223710  | 0.5146666666666670  | 0.521322137482435   | 4500.0             |

## Deep learning based classifer
### Count Plot
![DcountPlot](https://github.com/Amloner/Sentiment-Analysis-of-User-Reviews/assets/124287518/70885b53-25b2-41d1-9e12-d2e97c41c7f5)
### Box Plot
![DconfusionMatrix](https://github.com/Amloner/Sentiment-Analysis-of-User-Reviews/assets/124287518/f7ac2f26-78e7-4d7e-9d11-67e90c012326)
### Confusion Matrix
![DboxPlot](https://github.com/Amloner/Sentiment-Analysis-of-User-Reviews/assets/124287518/bd609ff0-c402-4025-b458-8a4ccf37d8e8)
### Report
|                  | **precision**       | **recall**         | **f1-score**       | **support**        |
| ---------------- | ------------------- | ------------------ | ------------------ | ------------------ |
| **negative**     | 0.7597955706984670  | 0.892              | 0.8206071757129720 | 2000.0             |
| **neutral**      | 0.21108742004264400 | 0.198              | 0.2043343653250770 | 500.0              |
| **positive**     | 0.9090909090909090  | 0.765              | 0.8308444203095300 | 2000.0             |
| **accuracy**     | 0.7584444444444450  | 0.7584444444444450 | 0.7584444444444450 | 0.7584444444444450 |
| **macro avg**    | 0.6266579666106730  | 0.6183333333333330 | 0.6185953204491930 | 4500.0             |
| **weighted avg** | 0.7651814821333500  | 0.7584444444444450 | 0.7566823054905650 | 4500.0             |
