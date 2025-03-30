# Here is an optional 'ESM2 (pML) linked to LR (Logistic Regression) Classifier model' training for detectability of noncanonical ORF microprotein sequences by Mass Spectrometry (Deutsch et al., 2024 https://pubmed.ncbi.nlm.nih.gov/39314370/)
# **ESM2 linked to LR Classifier Model**
-   1. Create a Google Colab file https://colab.research.google.com/#create=true (Google Account required)
    2. Once the file is created, under Notebook settings select T4 GPU as 'Hardware accelerator'
    3. Ensure you have the following Python libraries installed !pip install numpy pandas torch transformers tqdm scikit-learn matplotlib seaborn shap
    4. Copy the Python script from the 'ESM2_linked_to_LR_classifier_script.md' file and paste it on your Google Colab file
    5. Run the ESM2_linked_to_LR_classifier_script using the 'seq7264ORFs_dataset.csv' file as input ("/content/seq7264ORFs_dataset.csv") and visualize the results
-      See 'README_Detailed_microprotein_detectability_using_ESM2_and_LR_Classifier.md' for more details on the approach
-	   See the SHAP and the Permutation Importance Features plots showing the top variables of importance
-      The reader can expect a ROC AUC of 0.72 and PR AUC of 0.70 (depending on the random selection of test/train) 
