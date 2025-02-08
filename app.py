import streamlit as st
import pandas as pd
from main import OllamaRAG
import time

# Page configuration
st.set_page_config(
    page_title="RAG Chat System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize RAG system
@st.cache_resource
def init_rag():
    return OllamaRAG(performance=True)

rag = init_rag()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Chat", "Resources", "Conversation History", "Add Resource"])

# Chat Page
if page == "Chat":
    st.title("💬 Chat with RAG System")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask something..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            response, resources = rag.chat(prompt)
            message_placeholder.write(response)
            
            if resources:
                st.write("### Relevant Resources:")
                for resource in resources:
                    try:
                        # Create clickable link with proper syntax
                        st.markdown(f"[{resource['name']}]({resource['url']})")
                        
                        # Optional: Display metadata
                        with st.expander("Details"):
                            st.write(f"Description: {resource['description']}")
                            st.write(f"Tags: {resource['tags']}")
                    except Exception as e:
                        st.error(f"Error displaying resource: {str(e)}")
                        
            st.session_state.messages.append({"role": "assistant", "content": response})

# Resources Page
elif page == "Resources":
    st.title("📚 Resource Database")
    
    # Get and display resources
    resources = rag.db.get_all_resources()
    if resources:
        st.write("Total Resources:", len(resources))
        df = pd.DataFrame(resources, columns=["ID", "Name", "Description", "Content", "Created At", "Tags"])
        st.dataframe(df, use_container_width=True)
        
        # Resource details expander
        with st.expander("View Resource Details"):
            resource_id = st.number_input("Enter Resource ID", min_value=1, 
                                        max_value=len(resources))
            if st.button("Show Details"):
                resource = rag.db.get_resource(resource_id)
                if resource:
                    st.write("**Name:**", resource[1])
                    st.write("**Description:**", resource[2])
                    st.write("**Content:**", resource[3])
    else:
        st.info("No resources available yet.")

# Conversation History Page
elif page == "Conversation History":
    st.title("💭 Conversation History")
    
    conversations = rag.db.get_all_conversations()
    
    if conversations:
        st.write("Total Conversations:", len(conversations))
        # st.write(conversations)
        df = pd.DataFrame(conversations, 
                         columns=["ID", "User Input", "Assistant Response", "Created At"])
        st.dataframe(df, use_container_width=True)
        
        # Conversation details expander
        with st.expander("View Conversation Details"):
            conv_id = st.number_input("Enter Conversation ID", min_value=1, 
                                    max_value=len(conversations))
            if st.button("Show Details"):
                conv = rag.db.get_conversation(conv_id)
                if conv:
                    st.write("**User:**", conv[1])
                    st.write("**Assistant:**", conv[2])
    else:
        st.info("No conversations available yet.")

# Add Resource Page
elif page == "Add Resource":
    st.title("➕ Add New Resource")
    
    with st.form("add_resource_form"):
        name = st.text_input("Resource Name")
        content = st.text_area("Resource Content")
        
        if st.form_submit_button("Add Resource"):
            if name and content:
                with st.spinner("Adding resource..."):
                    try:
                        rag.add_resource(name, content)
                        st.success("Resource added successfully!")
                    except Exception as e:
                        st.error(f"Error adding resource: {str(e)}")
            else:
                st.warning("Please fill in both name and content fields.")

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ using Streamlit")