// --- Initial State ---
const initialState = {
    currentStep: 1,
    currentPrompt: "",
    conversationHistory: [], // Format: {role: 'user'/'assistant', content: '...'}
    isLoading: false,
    loadingMessage: "Processing...",
    errorMessage: null,
    isUserInputVisible: false,
    // Derived from UI elements in app.js
    initialPromptValue: "",
    userResponseValue: "",
    finalPromptContent: "",
    isFinalResultVisible: false,
    stepInfo: {
        title: "Initial Prompt Analysis", // Default for step 1
        desc: "Enter your initial prompt to begin the optimization process." // Default for step 1
    }
};

// --- Action Types ---
const ActionTypes = {
    SET_STEP: 'SET_STEP',
    SET_PROMPT: 'SET_PROMPT',
    SET_INITIAL_PROMPT_VALUE: 'SET_INITIAL_PROMPT_VALUE', // For the textarea
    ADD_CONVERSATION_MESSAGE: 'ADD_CONVERSATION_MESSAGE',
    CLEAR_CONVERSATION: 'CLEAR_CONVERSATION',
    SET_LOADING: 'SET_LOADING',
    SET_ERROR: 'SET_ERROR',
    CLEAR_ERROR: 'CLEAR_ERROR',
    SHOW_USER_INPUT: 'SHOW_USER_INPUT',
    SET_USER_RESPONSE_VALUE: 'SET_USER_RESPONSE_VALUE', // For the input field
    SET_FINAL_PROMPT: 'SET_FINAL_PROMPT',
    SHOW_FINAL_RESULT: 'SHOW_FINAL_RESULT',
    RESET_STATE: 'RESET_STATE',
    UPDATE_STEP_INFO: 'UPDATE_STEP_INFO'
};

// --- Action Creators ---
function setLoading(isLoading, message = "Processing...") {
    return { type: ActionTypes.SET_LOADING, payload: { isLoading, message } };
}

function setStep(step) {
    return { type: ActionTypes.SET_STEP, payload: step };
}

function setCurrentPrompt(prompt) {
    return { type: ActionTypes.SET_PROMPT, payload: prompt };
}

function setInitialPromptValue(value) {
    return { type: ActionTypes.SET_INITIAL_PROMPT_VALUE, payload: value };
}

function addConversationMessage(role, content) {
    return { type: ActionTypes.ADD_CONVERSATION_MESSAGE, payload: { role, content } };
}

function clearConversation() {
    return { type: ActionTypes.CLEAR_CONVERSATION };
}

function setError(message) {
    return { type: ActionTypes.SET_ERROR, payload: message };
}

function clearError() {
    return { type: ActionTypes.CLEAR_ERROR };
}

function showUserInput(isVisible) {
    return { type: ActionTypes.SHOW_USER_INPUT, payload: isVisible };
}

function setUserResponseValue(value) {
    return { type: ActionTypes.SET_USER_RESPONSE_VALUE, payload: value };
}

function setFinalPrompt(promptHTML) {
    return { type: ActionTypes.SET_FINAL_PROMPT, payload: promptHTML };
}

function showFinalResult(isVisible) {
    return { type: ActionTypes.SHOW_FINAL_RESULT, payload: isVisible };
}

function resetState() {
    return { type: ActionTypes.RESET_STATE };
}

function updateStepInfo(stepNumber, title, description) {
    // This could also be part of SET_STEP if step info is always derived from step number
    return { type: ActionTypes.UPDATE_STEP_INFO, payload: { stepNumber, title, description } };
}


// --- Root Reducer ---
function rootReducer(state = initialState, action) {
    switch (action.type) {
        case ActionTypes.SET_LOADING:
            return { ...state, isLoading: action.payload.isLoading, loadingMessage: action.payload.message };
        case ActionTypes.SET_STEP:
            return { ...state, currentStep: action.payload };
        case ActionTypes.SET_PROMPT:
            return { ...state, currentPrompt: action.payload };
        case ActionTypes.SET_INITIAL_PROMPT_VALUE:
            return { ...state, initialPromptValue: action.payload };
        case ActionTypes.ADD_CONVERSATION_MESSAGE:
            return { ...state, conversationHistory: [...state.conversationHistory, action.payload] };
        case ActionTypes.CLEAR_CONVERSATION:
            // Example: Keep system welcome message if desired, or truly clear
            return { ...state, conversationHistory: [initialState.conversationHistory[0] || {role: 'assistant', content: 'Welcome (default initial message)'}] };
        case ActionTypes.SET_ERROR:
            return { ...state, errorMessage: action.payload };
        case ActionTypes.CLEAR_ERROR:
            return { ...state, errorMessage: null };
        case ActionTypes.SHOW_USER_INPUT:
            return { ...state, isUserInputVisible: action.payload };
        case ActionTypes.SET_USER_RESPONSE_VALUE:
            return { ...state, userResponseValue: action.payload };
        case ActionTypes.SET_FINAL_PROMPT:
            return { ...state, finalPromptContent: action.payload };
        case ActionTypes.SHOW_FINAL_RESULT:
            return { ...state, isFinalResultVisible: action.payload };
        case ActionTypes.UPDATE_STEP_INFO:
            // This assumes stepInfo is an object like { title: '', desc: '' }
            // and that SET_STEP might also trigger a default update to this.
            return { ...state, stepInfo: { title: action.payload.title, desc: action.payload.description } };
        case ActionTypes.RESET_STATE:
            // Preserve conversation history or fully reset as needed.
            // For a full reset:
            return {
                ...initialState,
                // Potentially keep some things like API_BASE_URL if they were part of state
                // and not truly "resettable" UI state.
                // For this example, full reset to initial defined state.
                conversationHistory: [{ // Reset with a welcome message
                    role: 'assistant',
                    content: 'Welcome! I\'m here to help you optimize your prompts. Start by entering your initial prompt on the left, and I\'ll guide you through the refinement process.'
                }]
            };
        default:
            return state;
    }
}

// --- Create Store ---
const store = Redux.createStore(rootReducer);

// --- Expose to Window ---
window.store = store;
window.ActionTypes = ActionTypes;

// Expose action creators that will be called from app.js or HTML
window.setLoadingAction = setLoading;
window.setStepAction = setStep;
window.setCurrentPromptAction = setCurrentPrompt;
window.setInitialPromptValueAction = setInitialPromptValue;
window.addConversationMessageAction = addConversationMessage;
window.clearConversationAction = clearConversation;
window.setErrorAction = setError;
window.clearErrorAction = clearError;
window.showUserInputAction = showUserInput;
window.setUserResponseValueAction = setUserResponseValue;
window.setFinalPromptAction = setFinalPrompt;
window.showFinalResultAction = showFinalResult;
window.resetStateAction = resetState;
window.updateStepInfoAction = updateStepInfo;

// Dispatch initial actions if needed to set up the UI from defaults
// e.g. if conversation history needs a welcome message not hardcoded in initialState
if (initialState.conversationHistory.length === 0) {
    store.dispatch(addConversationMessage(
        'assistant',
        'Welcome! I\'m here to help you optimize your prompts. Start by entering your initial prompt on the left, and I\'ll guide you through the refinement process.'
    ));
}
store.dispatch(updateStepInfoAction(1, "Initial Prompt Analysis", "Enter your initial prompt to begin the optimization process."));
store.dispatch(setLoadingAction(false)); // Ensure loading is false initially
