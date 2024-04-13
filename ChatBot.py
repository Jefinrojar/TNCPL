import streamlit as st 
import google.generativeai as genai 
import google.ai.generativelanguage as glm 
from dotenv import load_dotenv
from PIL import Image
import os 
import io 
import requests


load_dotenv()

API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": "Bearer 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"}



def image_to_byte_array(image: Image) -> bytes:
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format=image.format)
    imgByteArr=imgByteArr.getvalue()
    return imgByteArr

API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

st.write("")


    
    
gemini_pro, gemini_vision = st.tabs(["CHATBOT", "CHAT WITH IMAGES"])


def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content

def main():
    
    
    with gemini_pro:
        
        st.header("Interact with SMART TUTOR 📚")
        st.write("")

        option = st.selectbox('Select the standard',
                              (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
        )   
        st.write('You selected:', option)
        
        option1 = st.selectbox('Select the Subject',
                              ("English", "Maths", "Science", "Social", "Tamil", "Chemistry", "Physics", "Biology", "zoology", "Botany")
        )   
        st.write('You selected:', option1)
        
        st.write("ask some questions from", option1)
        

        prompt = st.text_input("prompt please...", placeholder="Prompt", label_visibility="visible")
        model = genai.GenerativeModel("gemini-pro")
        
        if option<=5:
            if st.button("SEND",use_container_width=True):
                max_tokens = 50
                response = model.generate_content(prompt)
                generated_text = response.text
                tokens = generated_text.split()
                truncated_response = ' '.join(tokens[:max_tokens])
                st.write(truncated_response)
                if len(prompt)>1:
                    image_bytes = query({"inputs": prompt,})
                    image = Image.open(io.BytesIO(image_bytes))
                    st.image(image)
        
        elif option==6:
            if st.button("SEND",use_container_width=True):
                max_tokens = 100
                response = model.generate_content(prompt)
                generated_text = response.text
                tokens = generated_text.split()
                truncated_response = ' '.join(tokens[:max_tokens])
                st.write(truncated_response)
                if len(prompt)>1:
                    image_bytes = query({"inputs": prompt,})
                    image = Image.open(io.BytesIO(image_bytes))
                    st.image(image)
                    
        elif option==7:
            if st.button("SEND",use_container_width=True):
                max_tokens = 150
                response = model.generate_content(prompt)
                generated_text = response.text
                tokens = generated_text.split()
                truncated_response = ' '.join(tokens[:max_tokens])
                st.write(truncated_response)
                if len(prompt)>1:
                    image_bytes = query({"inputs": prompt,})
                    image = Image.open(io.BytesIO(image_bytes))
                    st.image(image)

        elif option==8:
            if st.button("SEND",use_container_width=True):
                max_tokens = 200
                response = model.generate_content(prompt)
                generated_text = response.text
                tokens = generated_text.split()
                truncated_response = ' '.join(tokens[:max_tokens])
                st.write(truncated_response)
                if len(prompt)>1:
                    image_bytes = query({"inputs": prompt,})
                    image = Image.open(io.BytesIO(image_bytes))
                    st.image(image)
        
        elif option==9:
            if st.button("SEND",use_container_width=True):
                max_tokens =  250
                response = model.generate_content(prompt)
                generated_text = response.text
                tokens = generated_text.split()
                truncated_response = ' '.join(tokens[:max_tokens])
                st.write(truncated_response)
                if len(prompt)>1:
                    image_bytes = query({"inputs": prompt,})
                    image = Image.open(io.BytesIO(image_bytes))
                    st.image(image)
                    
        elif option==10:
            if st.button("SEND",use_container_width=True):
                max_tokens = 300
                response = model.generate_content(prompt)
                generated_text = response.text
                tokens = generated_text.split()
                truncated_response = ' '.join(tokens[:max_tokens])
                st.write(truncated_response)
                if len(prompt)>1:
                    image_bytes = query({"inputs": prompt,})
                    image = Image.open(io.BytesIO(image_bytes))
                    st.image(image)
        
                    
        elif option<=12 and  option>10:
            if st.button("SEND",use_container_width=True):
                response = model.generate_content(prompt)
                st.header(":blue[Response]")
                st.write("")
                st.markdown(response.text)
                if len(prompt)>1:
                    image_bytes = query({"inputs": prompt,})
                    image = Image.open(io.BytesIO(image_bytes))
                    st.image(image)
            




    with gemini_vision:
        st.header("Interact with SMART TUTOR By giving images 🌃")
        st.write("")

        image_prompt = st.text_input("Interact with the Image", placeholder="Prompt", label_visibility="visible")
        uploaded_file = st.file_uploader("Choose and Image", accept_multiple_files=False, type=["png", "jpg", "jpeg", "img", "webp"])

        if uploaded_file is not None:
            st.image(Image.open(uploaded_file), use_column_width=True)

            st.markdown("""
                <style>
                        img {
                            border-radius: 10px;
                        }
                </style>
                """, unsafe_allow_html=True)
            
        if st.button("GET RESPONSE", use_container_width=True):
            model = genai.GenerativeModel("gemini-pro-vision")

            if uploaded_file is not None:
                if image_prompt != "":
                    image = Image.open(uploaded_file)

                    response = model.generate_content(
                        glm.Content(
                            parts = [
                                glm.Part(text=image_prompt),
                                glm.Part(
                                    inline_data=glm.Blob(
                                        mime_type="image/jpeg",
                                        data=image_to_byte_array(image)
                                    )
                                )
                            ]
                        )
                    )

                    response.resolve()

                    st.write("")
                    st.write(":blue[Response]")
                    st.write("")

                    st.markdown(response.text)

                else:
                    st.write("")
                    st.header(":red[Please Provide a prompt]")

            else:
                st.write("")
                st.header(":red[Please Provide an image]")




if __name__ == "__main__":
    main()
