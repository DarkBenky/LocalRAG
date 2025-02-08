import streamlit as st
import pandas as pd
from main import OllamaRAG, CODDER_MODEL, DEEP_SEEK_MODEL, CODDER_MODEL_BIG, CODDER_MODEL_SMALL, DEEP_SEEK_MODEL_BIG

# Page configuration
st.set_page_config(
    page_title="RAG Chat System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model" not in st.session_state:
    st.session_state.model = CODDER_MODEL_BIG
if "performance_mode" not in st.session_state:
    st.session_state.performance_mode = True
if "search_web" not in st.session_state:
    st.session_state.search_web = True
if "context_search" not in st.session_state:
    st.session_state.context_search = True

# Sidebar for model selection
with st.sidebar:
    st.title("⚙️ Settings")
    model_options = {
        "Qwen 3B": CODDER_MODEL,
        "Qwen 14B": CODDER_MODEL_BIG,
        "Qwen 1.5B": CODDER_MODEL_SMALL,
        "DeepSeek 32B": DEEP_SEEK_MODEL_BIG,
        "DeepSeek 1.5B": DEEP_SEEK_MODEL
    }
    selected_model = st.selectbox("Select Model", list(model_options.keys()))
    st.session_state.model = model_options[selected_model]

    # Fix: Store selectbox values directly in session state
    st.session_state.search_web = st.selectbox(
        "Search Web", 
        [True, False], 
        help="Enable or disable web search for resources."
    )
    
    st.session_state.performance_mode = st.selectbox(
        "Performance Mode", 
        [True, False], 
        help="Enable or disable performance mode for faster responses."
    )

    st.session_state.context_search = st.selectbox(
        "Context Search", 
        [True, False], 
        help="Enable or disable context search for better responses."
    )


# Initialize RAG with updated settings
rag = OllamaRAG(
    model_name=st.session_state.model,
    performance=st.session_state.performance_mode,
    web_search=st.session_state.search_web,
    context_search=st.session_state.context_search
)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Chat", "Resources", "Conversation History", "Add Resource"])

# Initialize chat state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.current_response = None

# Chat interface
if page == "Chat":
    st.title("💬 Chat")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "resources" in message:
                with st.expander("Referenced Resources"):
                    for resource in message["resources"]:
                        st.markdown(f"**{resource['name']}**")
                        st.write(f"Description: {resource.get('description', 'N/A')}")
    
    # Chat input
    prompt = st.chat_input("Ask something...")
    if prompt:
        # Clear previous response
        if st.session_state.get("current_response"):
            st.session_state.current_response = None
            
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
            
        try:
            # Generate new response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response, resources, info = rag.chat(prompt)
                    st.write(info)
                    st.write(response)
                    
                    if resources:
                        with st.expander("Referenced Resources"):
                            for resource in resources:
                                st.markdown(f"**{resource['name']}**")
                                st.write(f"URL: {resource.get('url', 'N/A')}")
                                st.write(f"Description: {resource.get('description', 'N/A')}")
                    
                    # Save response with resources
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "resources": resources
                    })
                    
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Resources Page
elif page == "Resources":
    st.title("📚 Resource Database")

    # Search functionality
    search_query = st.text_input("Search for a resource by name or description:")
    resources = rag.db.get_all_resources()

    if search_query:
        resources = [res for res in resources 
                     if search_query.lower() in res[1].lower() or search_query.lower() in res[2].lower()]

    if resources:
        st.write("Total Resources:", len(resources))
        df = pd.DataFrame(resources, columns=["ID", "Name", "Description", "Content", "Created At", "Tags"])
        st.dataframe(df, use_container_width=True)

        # Resource details expander
        with st.expander("View Resource Details"):
            resource_id = st.number_input("Enter Resource ID", min_value=1, max_value=len(resources))
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

    # Search functionality
    search_query = st.text_input("Search for a conversation by user input or assistant response:")
    conversations = rag.db.get_all_conversations()

    if search_query:
        conversations = [conv for conv in conversations 
                           if search_query.lower() in conv[1].lower() or search_query.lower() in conv[2].lower()]

    if conversations:
        st.write("Total Conversations:", len(conversations))
        df = pd.DataFrame(conversations, columns=["ID", "User Input", "Assistant Response", "Created At"])
        st.dataframe(df, use_container_width=True)

        # Conversation details expander
        with st.expander("View Conversation Details"):
            conv_id = st.number_input("Enter Conversation ID", min_value=1, max_value=len(conversations))
            if st.button("Show Details"):
                conv = rag.db.get_conversation(conv_id)
                if conv:
                    st.write("**User:**", conv[1])
                    st.write("**Assistant Response:**", conv[2])
    else:
        st.info("No conversations available yet.")

# Add Resource Page
elif page == "Add Resource":
    st.title("➕ Add New Resource")

    resource_name = st.text_input("Resource Name:")
    resource_description = st.text_area("Description:")
    resource_url = st.text_input("URL:")
    resource_tags = st.text_input("Tags (comma-separated):", help="Enter tags separated by commas.")

    if st.button("Add Resource"):
        try:
            tags = [tag.strip() for tag in resource_tags.split(",") if tag.strip()]
            rag.db.add_resource(name=resource_name, description=resource_description, url=resource_url, tags=tags)
            st.success("Resource added successfully!")
        except Exception as e:
            st.error(f"An error occurred while adding the resource: {str(e)}")

# Footer
st.markdown("---")
# st.write("Made with ❤️ by Your Na