# Further-treatment (YES/NO) decision: test set, n = 210

Cut-off rules chosen on validation: `sens80` = catch >= 80 % of reduced-EF ECGs, `youden` = max sensitivity + specificity, `max_acc` = max accuracy.

| Model                                                      |   Val AUC |   Test AUC |   sens80 acc |   sens80 sens |   sens80 spec |   youden acc |   youden sens |   youden spec |   max_acc acc |   max_acc sens |   max_acc spec |
|:-----------------------------------------------------------|----------:|-----------:|-------------:|--------------:|--------------:|-------------:|--------------:|--------------:|--------------:|---------------:|---------------:|
| Default: 4-class image + header fusion                     |     0.781 |      0.791 |        0.705 |         0.825 |         0.562 |        0.714 |         0.57  |         0.885 |         0.729 |          0.596 |          0.885 |
| 4-class image CNN (ResNet-18)                              |     0.777 |      0.735 |        0.671 |         0.719 |         0.615 |        0.695 |         0.632 |         0.771 |         0.7   |          0.64  |          0.771 |
| 4-class header model                                       |     0.751 |      0.795 |        0.695 |         0.825 |         0.542 |        0.676 |         0.561 |         0.812 |         0.676 |          0.561 |          0.812 |
| YES/NO header logistic regression                          |     0.767 |      0.807 |        0.69  |         0.833 |         0.521 |        0.733 |         0.632 |         0.854 |         0.733 |          0.632 |          0.854 |
| YES/NO waveform-feature logistic regression                |     0.754 |      0.728 |        0.638 |         0.746 |         0.51  |        0.7   |         0.57  |         0.854 |         0.676 |          0.588 |          0.781 |
| YES/NO waveform-feature XGBoost                            |     0.751 |      0.751 |        0.676 |         0.763 |         0.573 |        0.671 |         0.509 |         0.865 |         0.686 |          0.553 |          0.844 |
| YES/NO header logistic regression + image CNN (weight 0.7) |     0.789 |      0.789 |        0.705 |         0.789 |         0.604 |        0.719 |         0.64  |         0.812 |         0.719 |          0.64  |          0.812 |
| YES/NO waveform-feature XGBoost + image CNN (weight 0.0)   |     0.777 |      0.735 |        0.671 |         0.719 |         0.615 |        0.695 |         0.632 |         0.771 |         0.7   |          0.64  |          0.771 |
| YES/NO header 0.7 + waveform XGBoost 0.0 + image CNN 0.3   |     0.789 |      0.789 |        0.705 |         0.789 |         0.604 |        0.719 |         0.64  |         0.812 |         0.719 |          0.64  |          0.812 |
