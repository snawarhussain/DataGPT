🤖💬 **DataGPT**

DataGPT is a Streamlit web application that allows users to interact with advanced AI models like GPT, while leveraging the power of CSV data. Users can upload CSV files to
enhance the AI's knowledge and context, enabling more meaningful and data-driven conversations.

🌟 **Features**
- Chat interface with AI models like GPT from OpenAI, and LangChain
- Upload and utilize CSV files for more context-aware AI responses
- Clear and easy-to-use interface with separate containers for chat history and user input
- Runs it's own python code and uses the output to answer questions.
- Can provided statistics and other insights using Pandas


🚀 **Installation and Setup**
1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-chat-assistant.git
```
2. Navigate to the project directory:
```bash
cd DataGPT
```
3. Install the required packages:
```bash
pip install -r requirements.txt
```
4. Run the Streamlit app:
```bash
streamlit run app.py
```

🎯 **Usage**
1. Open the app in your web browser using the URL provided by Streamlit.
2. Select the AI model you want to use from the sidebar: OpenAI API or LangChain.
3. If you choose OpenAI API or LangChain, enter your OpenAI API key in the sidebar.
4. Upload a CSV file (optional) using the "Upload CSV File" section in the sidebar. The uploaded CSV data will be displayed on the main screen.
5. Type your message in the input field at the bottom of the main screen and press the "Send" button.
6. The AI's response will appear in the chat history container.

🤝 **Contributing**
Feel free to submit issues or pull requests to help improve the AI Chat Assistant app.

📄 **License**
This project is released under the MIT License.
