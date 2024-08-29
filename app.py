import streamlit as st
import pandas as pd
import torch
import pickle
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sentencex import segment
import os

# Load the model and tokenizer
@st.cache_resource
def load_model_and_tokenizer(output_dir):
    model = BertForSequenceClassification.from_pretrained(output_dir)
    tokenizer = BertTokenizer.from_pretrained(output_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, tokenizer, device

# Load the MultiLabelBinarizer
@st.cache_resource
def load_mlb(mlb_file):
    with open(mlb_file, 'rb') as f:
        mlb = pickle.load(f)
    return mlb

# Preprocess text
def preprocess_text(text, tokenizer, max_len=128):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    return encoding

# Predict labels
def predict(text, model, tokenizer, mlb, device, threshold, max_len=128):
    model.eval()  # Set the model to evaluation mode
    encoding = preprocess_text(text, tokenizer, max_len)
    input_ids = encoding['input_ids'].to(device)  # Move to device
    attention_mask = encoding['attention_mask'].to(device)  # Move to device

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.sigmoid(logits).cpu().numpy()

    predictions = (predictions >= threshold).astype(int)
    predicted_labels = mlb.inverse_transform(predictions)
    formatted_labels = '/'.join(label for label in predicted_labels[0])
    return formatted_labels

# Custom dataset class for training
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe['sentence'].tolist()
        self.labels = dataframe['labels'].tolist()
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = str(self.text[index])
        labels = self.labels[index]

        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'sentence': sentence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float)
        }

# Function to train the model
def train_model(train_df, test_df, mlb):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = CustomDataset(train_df, tokenizer, max_len=128)
    test_dataset = CustomDataset(test_df, tokenizer, max_len=128)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(mlb.classes_), problem_type="multi_label_classification")
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,  # Adjust epochs as needed
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
    )

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions >= 0.5  # For multi-label classification, apply thresholding

        return {
            'accuracy': accuracy_score(labels, preds),
            'f1': f1_score(labels, preds, average='micro'),
            'precision': precision_score(labels, preds, average='micro'),
            'recall': recall_score(labels, preds, average='micro'),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    results = trainer.evaluate()

    # Save the trained model
    output_dir = './saved_model'
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return results

# Streamlit app
def main():
    st.title("Text Classification with BERT")
    
    # Option to select between training a new model or using the existing model
    option = st.selectbox(
        "What would you like to do?",
        ("Test using the existing model", "Train a new model")
    )

    if option == "Test using the existing model":
        st.write("Upload a CSV file with a 'sentence' column or a TXT file to classify the segmented sentences.")

        # Slider for threshold
        threshold = st.slider("Set the classification threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

        # File uploader
        uploaded_file = st.file_uploader("Choose a CSV or TXT file", type=["csv", "txt"])

        if uploaded_file is not None:
            output_dir = './saved_model'  # Replace with your model directory
            model, tokenizer, device = load_model_and_tokenizer(output_dir)
            mlb_file = 'mlb.pkl'  # Path to the saved MultiLabelBinarizer
            mlb = load_mlb(mlb_file)

            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                df = df.dropna(subset=['sentence'])

                with st.spinner('Processing...'):
                    predicted_labels_list = []
                    for index, row in df.iterrows():
                        sentence = row['sentence']
                        predicted_labels = predict(sentence, model, tokenizer, mlb, device, threshold)
                        predicted_labels_list.append(predicted_labels)

                    df['predicted_labels'] = predicted_labels_list

                st.success('Processing complete!')
                st.write(df)

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(label="Download Predictions as CSV", data=csv, file_name='predicted_labels.csv', mime='text/csv')

            elif uploaded_file.name.endswith('.txt'):
                text_data = uploaded_file.read().decode('utf-8')
                segments = list(segment("en", text_data))

                with st.spinner('Processing...'):
                    results = []
                    for sentence in segments:
                        predicted_labels = predict(sentence, model, tokenizer, mlb, device, threshold)
                        results.append({'sentence': sentence, 'predicted_labels': predicted_labels})

                    df = pd.DataFrame(results)

                st.success('Processing complete!')
                st.write(df)

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(label="Download Predictions as CSV", data=csv, file_name='predicted_labels.csv', mime='text/csv')

    elif option == "Train a new model":
        st.write("Upload a CSV or TXT file to add to your training data and train a new model.")

        # File uploader
        uploaded_file = st.file_uploader("Choose a CSV or TXT file", type=["csv", "txt"])

        if uploaded_file is not None:
            # Convert TXT to CSV if necessary
            if uploaded_file.name.endswith('.txt'):
                text_data = uploaded_file.read().decode('utf-8')
                segments = list(segment("en", text_data))
                df = pd.DataFrame({'sentence': segments})
            else:
                df = pd.read_csv(uploaded_file)
                df = df.dropna(subset=['sentence'])

            # Check for existing training data
            existing_data_path = 'training_data.csv'
            if os.path.exists(existing_data_path):
                existing_df = pd.read_csv(existing_data_path)
                initial_sample_count = len(existing_df)
                df = pd.concat([existing_df, df], ignore_index=True)

                # Remove duplicates based on the 'sentence' column
                df.drop_duplicates(subset=['sentence'], inplace=True)
                
                st.write(f"Initial training data had {initial_sample_count} samples.")
                st.write(f"New training data has {len(df)} samples after appending and removing duplicates.")
            else:
                initial_sample_count = 0
                st.write("No existing training data found. Starting with the uploaded data.")

            # Save the updated dataset for future training
            df.to_csv(existing_data_path, index=False)

            # Preprocess labels
            df['labels'] = df['labels'].apply(lambda x: x.split('/'))
            mlb = MultiLabelBinarizer()
            df['labels'] = mlb.fit_transform(df['labels']).tolist()

            # Split the dataset into training and testing sets
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            train_df = train_df.reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)

            # Display the data
            st.write(df)

            # Training model section
            st.header("Train the Model")
            if st.button("Start Training"):
                st.write("Training in progress...")
                with st.spinner('Training the model...'):
                    results = train_model(train_df, test_df, mlb)

                st.success("Training complete!")
                st.write(f"Accuracy: {results['eval_accuracy']}")
                st.write(f"F1 Score: {results['eval_f1']}")
                st.write(f"Precision: {results['eval_precision']}")
                st.write(f"Recall: {results['eval_recall']}")

                # Save the MultiLabelBinarizer
                mlb_file = 'mlb.pkl'
                with open(mlb_file, 'wb') as f:
                    pickle.dump(mlb, f)

if __name__ == "__main__":
    main()