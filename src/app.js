// --- Global Constants & UI Elements ---
const API_BASE_URL = "http://127.0.0.1:5000"; // Backend API URL
const NO_QUESTION_MARKER = "NO_QUESTION";

const initialPromptTextarea = document.getElementById('initialPrompt');
const chatContainer = document.getElementById('chatContainer');
const userInputSection = document.getElementById('userInputSection');
const userResponseInput = document.getElementById('userResponse');
const finalResultCard = document.getElementById('finalResult');
const finalPromptContentEl = document.getElementById('finalPromptContent');
const stepInfoEl = document.getElementById('stepInfo');
const statusIndicatorEl = document.querySelector('.status-indicator');
const refineButton = document.querySelector('.btn-primary[onclick="startRefinement()"]');
const sendResponseButton = document.querySelector('.btn-secondary[onclick="sendUserResponse()"]');

const stepUIDefinitions = {
    1: { title: "Initial Prompt Analysis", desc: "Enter your initial prompt to begin the optimization process." },
    2: { title: "Prompt Refinement", desc: "AI analyzes and refines your prompt. Waiting for first clarifying question." },
    3: { title: "Clarifying Questions", desc: "Answer questions to further optimize your prompt." },
    4: { title: "Final Optimization", desc: "Receive your fully optimized prompt ready for use." }
};

// --- Redux Store Access ---
const store = window.store;
// Action creators are already on window object from store.js (e.g., window.setLoadingAction)

// --- Render Function ---
function renderApp() {
    const state = store.getState();

    // Update Step Indicators UI
    for (let i = 1; i <= 4; i++) {
        const stepEl = document.getElementById(`step${i}`);
        if (stepEl) {
            stepEl.classList.remove('active', 'completed');
            if (i < state.currentStep) stepEl.classList.add('completed');
            else if (i === state.currentStep) stepEl.classList.add('active');
        }
    }
    if (stepInfoEl && state.stepInfo) {
        stepInfoEl.innerHTML = `
            <strong>Step ${state.currentStep}:</strong> ${state.stepInfo.title}
            <p style="margin-top: 5px; color: #666; font-size: 0.9rem;">${state.stepInfo.desc}</p>`;
    }

    // Update Loading State UI
    if (statusIndicatorEl) {
        if (state.isLoading) {
            statusIndicatorEl.className = 'status-indicator status-processing loading';
            statusIndicatorEl.innerHTML = `<i class="fas fa-spinner fa-spin"></i> ${state.loadingMessage}`;
        } else {
            // Display "Waiting for your response" if applicable, otherwise "System Ready"
            const isWaitingForUser = state.isUserInputVisible && state.currentStep === 3; // Example condition
            if (isWaitingForUser) {
                 statusIndicatorEl.className = 'status-indicator status-ready'; // Or a different class
                 statusIndicatorEl.innerHTML = `<i class="fas fa-question-circle"></i> Waiting for your response`;
            } else {
                statusIndicatorEl.className = 'status-indicator status-ready';
                statusIndicatorEl.innerHTML = `<i class="fas fa-check-circle"></i> System Ready`;
            }
        }
    }
    if(refineButton) refineButton.disabled = state.isLoading;
    if(sendResponseButton) sendResponseButton.disabled = state.isLoading;

    // Update Chat Messages UI
    if (chatContainer) {
        chatContainer.innerHTML = ''; // Clear existing messages
        state.conversationHistory.forEach(msg => {
            const messageDiv = document.createElement('div');
            const validRoles = ['user', 'assistant', 'error', 'system-error'];
            const messageRole = validRoles.includes(msg.role) ? msg.role : 'assistant';
            messageDiv.className = `chat-message ${messageRole}`;

            const bubble = document.createElement('div');
            bubble.className = 'message-bubble';
            bubble.innerHTML = msg.content.replace(/\n/g, '<br>');

            messageDiv.appendChild(bubble);
            chatContainer.appendChild(messageDiv);
        });
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    // Update User Input Section UI
    if (userInputSection) userInputSection.classList.toggle('hidden', !state.isUserInputVisible);
    if (userResponseInput) {
        if(state.isUserInputVisible && document.activeElement !== userResponseInput) {
            // Only focus if not already focused, to avoid disrupting user typing
            userResponseInput.focus();
        }
        // Keep input value in sync with state (e.g., when cleared after send)
        if (userResponseInput.value !== state.userResponseValue) {
            userResponseInput.value = state.userResponseValue;
        }
    }

    // Update Initial Prompt Textarea (if it were controlled by Redux state)
    // For now, initialPromptTextarea is read directly by event handlers.
    // if (initialPromptTextarea && initialPromptTextarea.value !== state.initialPromptValue) {
    //      initialPromptTextarea.value = state.initialPromptValue;
    // }

    // Update Final Result UI
    if (finalResultCard) finalResultCard.classList.toggle('hidden', !state.isFinalResultVisible);
    if (finalPromptContentEl) finalPromptContentEl.innerHTML = state.finalPromptContent || "";

    // Error messages are now part of conversationHistory with role 'error'.
    // No need for separate handling here if store.js/reducers add errors to history.
    // If a global, non-chat error display is desired, it would be handled here.
}

// --- Subscribe and Initial Render ---
store.subscribe(renderApp);
renderApp(); // Initial render

// --- Refactored Action-Dispatching Functions (no direct DOM manipulation) ---
function dispatchUpdateStepUI(step) {
    const newStepInfo = stepUIDefinitions[step] || { title: "Unknown Step", desc: "" };
    store.dispatch(window.updateStepInfoAction(step, newStepInfo.title, newStepInfo.desc));
    store.dispatch(window.setStepAction(step));
}

function dispatchShowUserInput(isVisible) {
    store.dispatch(window.showUserInputAction(isVisible));
    if (!isVisible) {
        store.dispatch(window.setUserResponseValueAction("")); // Clear input value when hiding
    }
}

// --- HTML Event Handlers (Fully Redux Integrated) ---
window.useExamplePrompt = function(element) {
    const promptText = element.querySelector('p').textContent;
    if (initialPromptTextarea) initialPromptTextarea.value = promptText;
    // Dispatch if this value needs to be in Redux state for other reasons
    // store.dispatch(window.setInitialPromptValueAction(promptText));
}

window.startRefinement = async function() {
    const initialPromptValue = initialPromptTextarea ? initialPromptTextarea.value.trim() : "";
    if (!initialPromptValue) {
        alert('Please enter an initial prompt first.'); // Simple validation, could be a Redux state error too
        return;
    }

    store.dispatch(window.clearErrorAction());
    dispatchUpdateStepUI(2);
    store.dispatch(window.addConversationMessageAction('user', `Initial prompt: "${initialPromptValue}"`));
    store.dispatch(window.setLoadingAction(true, "Refining prompt..."));
    store.dispatch(window.setCurrentPromptAction(initialPromptValue));

    try {
        const response = await fetch(`${API_BASE_URL}/refine_initial_prompt`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ initial_prompt: initialPromptValue })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        store.dispatch(window.setCurrentPromptAction(data.refined_prompt));

        // Replace conversation history with that from the backend
        store.dispatch(window.clearConversationAction());
        data.conversation_history.forEach(msg => {
            store.dispatch(window.addConversationMessageAction(msg.role, msg.content));
        });
        // The refined prompt itself is usually part of the conversation history from backend
        // If not, add it: store.dispatch(window.addConversationMessageAction('assistant', `**Refined Prompt:**<br>${data.refined_prompt}`));

        store.dispatch(window.setLoadingAction(false));
        await fetchClarifyingQuestion();

    } catch (error) {
        console.error("Error in startRefinement:", error);
        store.dispatch(window.setErrorAction(`Failed to refine prompt: ${error.message || 'Unknown error'}`));
        store.dispatch(window.addConversationMessageAction('error', `Failed to refine prompt: ${error.message || 'Unknown error'}`));
        store.dispatch(window.setLoadingAction(false));
        dispatchUpdateStepUI(1); // Revert to step 1
    }
}

async function fetchClarifyingQuestion() {
    const state = store.getState(); // Get current state for API call
    dispatchUpdateStepUI(3);
    store.dispatch(window.setLoadingAction(true, "Fetching clarifying question..."));
    dispatchShowUserInput(false);
    store.dispatch(window.clearErrorAction());

    try {
        const response = await fetch(`${API_BASE_URL}/get_clarifying_question`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                current_prompt: state.currentPrompt,
                conversation_history: state.conversationHistory
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        // Replace conversation history
        store.dispatch(window.clearConversationAction());
        data.updated_conversation_history.forEach(msg => {
            store.dispatch(window.addConversationMessageAction(msg.role, msg.content));
        });

        if (data.question && data.question.trim().toUpperCase() !== NO_QUESTION_MARKER) {
            dispatchShowUserInput(true);
            // Status indicator update is handled by renderApp based on isLoading and other state
        } else {
            // If no question, the last message in history might be "No further questions..."
            // Or add it explicitly if backend doesn't:
            // store.dispatch(window.addConversationMessageAction('assistant', "No further questions. Generating final prompt..."));
            await generateFinalPromptFromServer();
        }
    } catch (error) {
        console.error("Error in fetchClarifyingQuestion:", error);
        store.dispatch(window.setErrorAction(`Failed to fetch clarifying question: ${error.message || 'Unknown error'}`));
        store.dispatch(window.addConversationMessageAction('error', `Failed to fetch clarifying question: ${error.message || 'Unknown error'}`));
    } finally {
        store.dispatch(window.setLoadingAction(false));
    }
}

window.sendUserResponse = async function() {
    const state = store.getState();
    const userAnswer = state.userResponseValue.trim(); // Get value from state
    if (!userAnswer) return;

    store.dispatch(window.addConversationMessageAction('user', userAnswer));
    store.dispatch(window.setUserResponseValueAction("")); // Clear input field via state
    store.dispatch(window.setLoadingAction(true, "Processing your answer..."));
    dispatchShowUserInput(false);
    store.dispatch(window.clearErrorAction());

    // Get latest state for API call, especially conversationHistory
    const currentState = store.getState();

    try {
        const response = await fetch(`${API_BASE_URL}/submit_answer_and_get_next`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                answer: userAnswer,
                current_prompt: currentState.currentPrompt,
                conversation_history: currentState.conversationHistory
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        store.dispatch(window.clearConversationAction());
        data.updated_conversation_history.forEach(msg => {
            store.dispatch(window.addConversationMessageAction(msg.role, msg.content));
        });
        store.dispatch(window.setCurrentPromptAction(data.current_prompt_after_answer || currentState.currentPrompt));

        if (data.type === "question") {
            if (data.question && data.question.trim().toUpperCase() !== NO_QUESTION_MARKER) {
                dispatchShowUserInput(true);
            } else {
                // store.dispatch(window.addConversationMessageAction('assistant', "No further questions. Generating final prompt..."));
                await generateFinalPromptFromServer();
            }
        } else if (data.type === "final_prompt") {
            dispatchUpdateStepUI(4);
            store.dispatch(window.setFinalPromptAction(data.final_prompt.replace(/\n/g, '<br>')));
            store.dispatch(window.showFinalResultAction(true));
            // Scrolling handled by renderApp or a dedicated UI effect function if needed
            if (finalResultCard) finalResultCard.scrollIntoView({ behavior: 'smooth' });
        }
    } catch (error) {
        console.error("Error in sendUserResponse:", error);
        store.dispatch(window.setErrorAction(`Failed to process your answer: ${error.message || 'Unknown error'}`));
        store.dispatch(window.addConversationMessageAction('error', `Failed to process your answer: ${error.message || 'Unknown error'}`));
    } finally {
        store.dispatch(window.setLoadingAction(false));
    }
}

async function generateFinalPromptFromServer() {
    const state = store.getState();
    // store.dispatch(window.addConversationMessageAction('assistant', "Attempting to generate final prompt..."));
    store.dispatch(window.setLoadingAction(true, "Generating final prompt..."));
    store.dispatch(window.clearErrorAction());

    try {
         const response = await fetch(`${API_BASE_URL}/submit_answer_and_get_next`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                answer: "NO_QUESTION_PROCEED_TO_FINALIZE",
                current_prompt: state.currentPrompt,
                conversation_history: state.conversationHistory
            })
        });
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }
        const data = await response.json();

        store.dispatch(window.clearConversationAction());
        data.updated_conversation_history.forEach(msg => {
            store.dispatch(window.addConversationMessageAction(msg.role, msg.content));
        });

        if (data.type === "final_prompt") {
            dispatchUpdateStepUI(4);
            store.dispatch(window.setFinalPromptAction(data.final_prompt.replace(/\n/g, '<br>')));
            store.dispatch(window.showFinalResultAction(true));
            if (finalResultCard) finalResultCard.scrollIntoView({ behavior: 'smooth' });
        } else if (data.type === "question") {
             store.dispatch(window.addConversationMessageAction('error', `Unexpectedly received another question: ${data.question}`));
             dispatchShowUserInput(true);
        }
    } catch (error) {
        console.error("Error in generateFinalPromptFromServer:", error);
        store.dispatch(window.setErrorAction(`Failed to generate final prompt: ${error.message || 'Unknown error'}`));
        store.dispatch(window.addConversationMessageAction('error', `Failed to generate final prompt: ${error.message || 'Unknown error'}`));
    } finally {
        store.dispatch(window.setLoadingAction(false));
    }
}

window.copyToClipboard = function() {
    const state = store.getState();
    // Create a temporary element to parse HTML entities from finalPromptContent
    const tempEl = document.createElement('div');
    tempEl.innerHTML = state.finalPromptContent;
    const finalPromptText = tempEl.textContent || tempEl.innerText || "";

    navigator.clipboard.writeText(finalPromptText).then(() => {
        const btn = event.target.closest('button');
        if (btn) {
            const originalText = btn.innerHTML;
            btn.innerHTML = '<i class="fas fa-check"></i> Copied!';
            btn.style.background = '#48bb78';
            setTimeout(() => {
                btn.innerHTML = originalText;
                btn.style.background = '';
            }, 2000);
        }
    }).catch(err => {
        console.error('Failed to copy text: ', err);
        store.dispatch(window.setErrorAction('Failed to copy text. Please try again or copy manually.'));
        store.dispatch(window.addConversationMessageAction('error', 'Failed to copy text. Please try again or copy manually.'));
    });
}

window.startOver = function() {
    store.dispatch(window.resetStateAction());
    if (initialPromptTextarea) initialPromptTextarea.value = '';
    // userResponseInput value is cleared by renderApp based on state.isUserInputVisible and state.userResponseValue
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Event Listeners for controlled inputs
if (userResponseInput) {
    userResponseInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            window.sendUserResponse();
        }
    });
    userResponseInput.addEventListener('input', (e) => {
        store.dispatch(window.setUserResponseValueAction(e.target.value));
    });
}
// initialPromptTextarea is read directly on useExamplePrompt and startRefinement for now.
// If it needed to be fully controlled by Redux:
// if (initialPromptTextarea) {
//     initialPromptTextarea.addEventListener('input', (e) => {
//         store.dispatch(window.setInitialPromptValueAction(e.target.value));
//     });
// }

document.addEventListener('DOMContentLoaded', () => {
    renderApp(); // Re-render just in case store dispatches happened before DOM was fully ready
    console.log("App fully refactored and initialized with Redux store.");
});
