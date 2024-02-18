import streamlit as st
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nepali_unicode_converter.convert import Converter

# Load the model and tokenizer
model = load_model('next_words.h5')
tokenizer = pickle.load(open('token.pkl', 'rb'))

# Nepali Unicode Converter
converter = Converter()

def convert_to_nepali(word):
    return converter.convert(word)

def Predict_Next_Words(model, tokenizer, text, top_n=5):
    sequence_data = tokenizer.texts_to_sequences([text])[0]
    sequence = pad_sequences([sequence_data], maxlen=3, padding='pre', truncating='post')[0]
    sequence = np.array(sequence)

    preds = model.predict(sequence.reshape(1, -1))
    top_preds_indices = np.argsort(preds[0])[::-1][:top_n]

    predicted_words = []

    for index in top_preds_indices:
        for key, value in tokenizer.word_index.items():
            if value == index:
                predicted_word = key
                predicted_words.append(predicted_word)
                break

    return predicted_words

def main():
    st.title("Nepali Word Flow")

    # Create a session state to store the last input
    if "last_input" not in st.session_state:
        st.session_state.last_input = ""

    # Use st.sidebar to create a sidebar layout
    sidebar = st.sidebar

    # Add title "Mapping Table" to the sidebar
    sidebar.title("Mapping Table")

    # Display guide images in the sidebar with increased width
    sidebar.image("01.PNG", use_column_width=True, width=300)
    sidebar.image("02.PNG", use_column_width=True, width=300)
    sidebar.image("03.PNG", use_column_width=True, width=300)

    # Use st.columns to create a two-column layout in the main area
    col1, col2 = st.columns(2)

    # Reduce the gap between the sidebar and the input area
    st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            padding-top: 0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Use st.text_area for both text areas
    user_input = col1.text_area("Enter your text:", key="user_input", height=300)

    try:
        # Convert the entire text to Nepali
        converted_text = convert_to_nepali(user_input)

        # Display the converted text using st.text_area
        converted_text = col2.text_area("Converted Text:", value=converted_text, height=300)

        # Use on_change event to trigger space bar press
        if (
            st.session_state.last_input != converted_text
            and converted_text.endswith(" ")
        ):
            # Update the session state with the current input
            st.session_state.last_input = converted_text

            # Predict the next words after the entire text is converted
            predictions_after_conversion = Predict_Next_Words(model, tokenizer, st.session_state.last_input, top_n=5)

            # Display the predicted suggestions after conversion
            col1.text("Predicted Suggestions:")
            for predicted_word_after_conversion in predictions_after_conversion:
                # Convert the predicted word to Nepali
                converted_predicted_word = convert_to_nepali(predicted_word_after_conversion)
                col1.text(converted_predicted_word)

    except Exception as e:
        st.error(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
