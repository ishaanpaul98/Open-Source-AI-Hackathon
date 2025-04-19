import streamlit as st
from agent import Agent

def main():
    # Initialize session state variables
    if "agent" not in st.session_state:
        st.session_state.agent = None
        st.session_state.brief = ""
        st.session_state.budget = 100
        st.session_state.priority = "balanced"
    
    if "feedback" not in st.session_state:
        st.session_state.feedback = ""

    # Add custom CSS for dynamic background
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    
    @keyframes gradient {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }
    
    /* Style the main content area */
    .main .block-container {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 2rem;
        margin-top: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Style the sidebar */
    .css-1d391kg {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Style buttons */
    .stButton>button {
        background: linear-gradient(45deg, #2196F3, #21CBF3);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Style text inputs */
    .stTextArea>div>div>textarea,
    .stNumberInput>div>div>input {
        background-color: rgba(240, 240, 240, 0.95);
        border-radius: 5px;
        border: 1px solid #ddd;
        color: #333;
    }
    
    /* Style disabled text areas */
    .stTextArea>div>div>textarea:disabled {
        background-color: rgba(220, 220, 220, 0.95);
        color: #666;
    }
    
    /* Style headers */
    h1, h2, h3 {
        color: #333;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("AI Personal Shopper")
    
    # Sidebar for project details
    with st.sidebar:
        st.header("What would you like to shop for today?")
        brief = st.text_area("Project Brief", 
                           st.session_state.brief or "What would you like to shop for today?",
                           help="Describe your project requirements in detail")
        
        budget = st.number_input("Budget ($)", 
                               min_value=1, 
                               max_value=100000, 
                               value=st.session_state.budget,
                               step=100,
                               help="Enter your total budget in dollars")
        
        priority = st.selectbox("Priority",
                              ["balanced", "performance", "aesthetic"],
                              index=["balanced", "performance", "aesthetic"].index(st.session_state.priority),
                              help="Select your main priority for the project")
    
    # Main content area
    if st.button("Start Analysis"):
        # Initialize agent
        st.session_state.agent = Agent(brief=brief, budget=budget, priority=priority)
        st.session_state.brief = brief
        st.session_state.budget = budget
        st.session_state.priority = priority
        
        # Prepare initial state
        initial_state = {
            'messages': st.session_state.agent.messages,
            'brief': st.session_state.agent.state.brief,
            'budget': st.session_state.agent.state.budget,
            'priority': st.session_state.agent.state.priority,
            'analysis': {},
            'rethink': {},
            'iteration': 0
        }
        
        # Create a container for the analysis
        analysis_container = st.container()
        
        try:
            # Run the workflow
            final_state = st.session_state.agent.app.invoke(initial_state)
            
            # Update agent state
            st.session_state.agent.messages = final_state.get('messages', [])
            
            # Get results
            analysis = final_state.get('analysis', {})
            rethink = final_state.get('rethink', {})
            iterations = final_state.get('iteration', 0)
            
            # Display analysis
            with analysis_container:
                st.header("Project Analysis")
                
                if 'error' in analysis:
                    st.error(f"Error during analysis: {analysis['error']}")
                else:
                    # Summary
                    st.metric("Total Estimated Cost", f"${analysis.get('total_estimated_cost', 'N/A')}")
                    
                    # Categories and Items
                    st.subheader("Categories and Items")
                    for category in analysis.get('categories', []):
                        with st.expander(category.get('name', 'Unnamed Category')):
                            for item in category.get('items', []):
                                st.markdown(f"""
                                **{item.get('name', 'Unnamed Item')}** (${item.get('estimated_cost', 'N/A')})
                                - Priority: {item.get('priority', 'N/A')}
                                - Notes: {item.get('notes', 'N/A')}
                                """)
                
                # Display review
                st.header("Analysis Review")
                if 'error' in rethink:
                    st.error(f"Error during review: {rethink['error']}")
                else:
                    # Review metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Alignment Score", f"{rethink.get('alignment_score', 'N/A')}/100")
                    with col2:
                        st.markdown(f"**Overall Feedback:** {rethink.get('overall_feedback', 'N/A')}")
                    
                    # Recommendations
                    if rethink.get('recommendations'):
                        st.subheader("Recommendations")
                        for rec in rethink['recommendations']:
                            with st.expander(f"{rec.get('category', 'N/A')} - {rec.get('item', 'N/A')}"):
                                st.markdown(f"""
                                **Suggestion:** {rec.get('suggestion', 'N/A')}
                                
                                **Expected Impact:** {rec.get('impact', 'N/A')}
                                """)
        
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")

    # Feedback section (moved outside the Start Analysis button)
    if st.session_state.agent is not None:
        st.header("Feedback")
        feedback = st.text_area("Provide feedback to improve the analysis", 
                              value=st.session_state.feedback,
                              help="Your feedback will be incorporated into the next analysis")
        
        submit_feedback = st.button("Submit Feedback")
        if submit_feedback:
            print("Feedback button clicked")  # Debug message in terminal
            if feedback.strip():
                print("Feedback is not empty")  # Debug message in terminal
                # Update the brief with feedback
                updated_brief = f"{st.session_state.brief}\n\nUser Feedback: {feedback}"
                print(f"Updated brief: {updated_brief}")  # Debug message in terminal
                
                # Update session state
                st.session_state.brief = updated_brief
                st.session_state.feedback = feedback
                
                # Create new agent with updated brief
                st.session_state.agent = Agent(
                    brief=updated_brief, 
                    budget=st.session_state.budget, 
                    priority=st.session_state.priority
                )
                
                # Update the agent's state directly
                st.session_state.agent.state.brief = updated_brief
                
                # Prepare new state
                new_state = {
                    'messages': st.session_state.agent.messages,
                    'brief': updated_brief,
                    'budget': st.session_state.agent.state.budget,
                    'priority': st.session_state.agent.state.priority,
                    'analysis': {},
                    'rethink': {},
                    'iteration': 0,
                    'user_feedback': feedback
                }
                
                # Run new analysis
                new_final_state = st.session_state.agent.app.invoke(new_state)
                
                # Update the current state with new results
                analysis = new_final_state.get('analysis', {})
                rethink = new_final_state.get('rethink', {})
                
                # Clear the container and display new results
                analysis_container.empty()
                with analysis_container:
                    st.header("Project Analysis")
                    
                    if 'error' in analysis:
                        st.error(f"Error during analysis: {analysis['error']}")
                    else:
                        # Summary
                        st.metric("Total Estimated Cost", f"${analysis.get('total_estimated_cost', 'N/A')}")
                        
                        # Categories and Items
                        st.subheader("Categories and Items")
                        for category in analysis.get('categories', []):
                            with st.expander(category.get('name', 'Unnamed Category')):
                                for item in category.get('items', []):
                                    st.markdown(f"""
                                    **{item.get('name', 'Unnamed Item')}** (${item.get('estimated_cost', 'N/A')})
                                    - Priority: {item.get('priority', 'N/A')}
                                    - Notes: {item.get('notes', 'N/A')}
                                    """)
                    
                    # Display review
                    st.header("Analysis Review")
                    if 'error' in rethink:
                        st.error(f"Error during review: {rethink['error']}")
                    else:
                        # Review metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Alignment Score", f"{rethink.get('alignment_score', 'N/A')}/100")
                        with col2:
                            st.markdown(f"**Overall Feedback:** {rethink.get('overall_feedback', 'N/A')}")
                        
                        # Recommendations
                        if rethink.get('recommendations'):
                            st.subheader("Recommendations")
                            for rec in rethink['recommendations']:
                                with st.expander(f"{rec.get('category', 'N/A')} - {rec.get('item', 'N/A')}"):
                                    st.markdown(f"""
                                    **Suggestion:** {rec.get('suggestion', 'N/A')}
                                    
                                    **Expected Impact:** {rec.get('impact', 'N/A')}
                                    """)
                    
                    # Show the updated brief
                    st.header("Updated Project Brief")
                    st.text_area("Current Brief", 
                               value=st.session_state.brief,
                               height=200,
                               disabled=True)
                    
                    # Force a rerun to update the UI
                    st.experimental_rerun()
            else:
                st.warning("Please provide feedback before submitting")

if __name__ == "__main__":
    main() 