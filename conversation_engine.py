# File: conversation_engine.py
# Text response generation for audio conversations

import logging
import random
import re
from typing import Optional, Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class ConversationResponseGenerator:
    """
    Generates text responses for audio conversations.
    
    Supports multiple response modes:
    - echo: Simple echo/repeat of input
    - template: Template-based responses with keyword matching
    - (future: llm integration)
    """
    
    def __init__(self, response_mode: str = "echo"):
        """
        Initialize response generator.
        
        Args:
            response_mode: Response generation mode ("echo", "template")
        """
        self.response_mode = response_mode
        self.conversation_history = []
        self.template_responses = self._load_default_templates()
        
        logger.info(f"ConversationResponseGenerator initialized with mode: {response_mode}")
    
    def generate_response(self, input_text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a text response based on input text and mode.
        
        Args:
            input_text: Transcribed text from user
            context: Additional context (user info, conversation state, etc.)
            
        Returns:
            Generated response text
        """
        if not input_text or not input_text.strip():
            return self._get_fallback_response()
        
        # Clean and normalize input
        cleaned_input = self._clean_text(input_text)
        
        # Add to conversation history
        self.conversation_history.append({
            'timestamp': datetime.now(),
            'input': cleaned_input,
            'context': context or {}
        })
        
        # Keep history manageable
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-25:]
        
        # Generate response based on mode
        if self.response_mode == "echo":
            response = self._generate_echo_response(cleaned_input)
        elif self.response_mode == "template":
            response = self._generate_template_response(cleaned_input, context)
        else:
            logger.warning(f"Unknown response mode: {self.response_mode}")
            response = self._get_fallback_response()
        
        logger.debug(f"Generated response for '{cleaned_input[:50]}...': '{response[:50]}...'")
        return response
    
    def _generate_echo_response(self, input_text: str) -> str:
        """Generate echo-style responses."""
        echo_patterns = [
            f"I heard you say: {input_text}",
            f"You said: {input_text}",
            f"Did you say: {input_text}?",
            f"I understand you said: {input_text}",
            f"Let me repeat that: {input_text}",
        ]
        
        # Add some variety based on input length
        if len(input_text) < 10:
            echo_patterns.extend([
                f"'{input_text}' - got it!",
                f"I heard '{input_text}'",
            ])
        
        return random.choice(echo_patterns)
    
    def _generate_template_response(self, input_text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate template-based responses using keyword matching."""
        input_lower = input_text.lower()
        
        # Check for matches in template categories
        for category, templates in self.template_responses.items():
            if self._matches_category(input_lower, category):
                response_template = random.choice(templates['responses'])
                
                # Simple template variable substitution
                response = self._substitute_template_vars(response_template, input_text, context)
                return response
        
        # No specific template match - use generic responses
        generic_responses = self.template_responses.get('generic', {}).get('responses', [])
        if generic_responses:
            response_template = random.choice(generic_responses)
            return self._substitute_template_vars(response_template, input_text, context)
        
        # Fallback to echo if no templates available
        return self._generate_echo_response(input_text)
    
    def _matches_category(self, input_text: str, category: str) -> bool:
        """Check if input text matches a template category."""
        category_data = self.template_responses.get(category, {})
        keywords = category_data.get('keywords', [])
        patterns = category_data.get('patterns', [])
        
        # Check keyword matches
        for keyword in keywords:
            if keyword.lower() in input_text:
                return True
        
        # Check regex pattern matches
        for pattern in patterns:
            try:
                if re.search(pattern, input_text, re.IGNORECASE):
                    return True
            except re.error:
                logger.warning(f"Invalid regex pattern: {pattern}")
        
        return False
    
    def _substitute_template_vars(self, template: str, input_text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Substitute variables in response templates."""
        context = context or {}
        
        # Available template variables
        substitutions = {
            '{input}': input_text,
            '{time}': datetime.now().strftime("%H:%M"),
            '{date}': datetime.now().strftime("%Y-%m-%d"),
            '{user}': context.get('user_name', 'there'),
            '{count}': str(len(self.conversation_history)),
        }
        
        # Perform substitutions
        result = template
        for var, value in substitutions.items():
            result = result.replace(var, value)
        
        return result
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize input text."""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common transcription artifacts
        cleaned = re.sub(r'\b(um|uh|er|ah)\b', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def _get_fallback_response(self) -> str:
        """Get a fallback response when generation fails."""
        fallbacks = [
            "I'm sorry, I didn't catch that.",
            "Could you please repeat that?",
            "I'm not sure what you said.",
            "Can you say that again?",
            "I didn't understand that.",
        ]
        return random.choice(fallbacks)
    
    def _load_default_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Load default response templates."""
        return {
            'greeting': {
                'keywords': ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening'],
                'patterns': [r'\b(hi|hello|hey)\b'],
                'responses': [
                    "Hello {user}! How are you doing?",
                    "Hi there! Nice to hear from you!",
                    "Hey! What's up?",
                    "Hello! How can I help you today?",
                    "Hi! Great to chat with you!",
                ]
            },
            'farewell': {
                'keywords': ['bye', 'goodbye', 'see you', 'farewell', 'talk later'],
                'patterns': [r'\b(bye|goodbye|see you)\b'],
                'responses': [
                    "Goodbye {user}! Take care!",
                    "See you later! Have a great day!",
                    "Bye! Thanks for chatting!",
                    "Farewell! Until next time!",
                    "Take care! Talk to you soon!",
                ]
            },
            'question_how': {
                'keywords': ['how are you', 'how do you', 'how is', 'how was'],
                'patterns': [r'\bhow\s+(are|do|is|was)\b'],
                'responses': [
                    "I'm doing well, thank you for asking! How about you?",
                    "I'm great! Thanks for checking in. How are you?",
                    "I'm doing fine! How has your day been?",
                    "All good here! What about you?",
                    "I'm well, thanks! How are things with you?",
                ]
            },
            'question_what': {
                'keywords': ['what is', 'what are', 'what do', 'what time'],
                'patterns': [r'\bwhat\s+(is|are|do|time)\b'],
                'responses': [
                    "That's an interesting question about {input}",
                    "Hmm, let me think about that...",
                    "Good question! I'm not sure about that specific thing.",
                    "You're asking about something complex there.",
                    "That's worth exploring further.",
                ]
            },
            'affirmation': {
                'keywords': ['yes', 'yeah', 'yep', 'sure', 'okay', 'alright', 'absolutely'],
                'patterns': [r'\b(yes|yeah|yep|sure|okay|alright)\b'],
                'responses': [
                    "Great! I'm glad we're on the same page.",
                    "Awesome! Let's keep going.",
                    "Perfect! That sounds good.",
                    "Excellent! I agree.",
                    "Wonderful! That works for me.",
                ]
            },
            'negation': {
                'keywords': ['no', 'nope', 'not really', 'i disagree', 'wrong'],
                'patterns': [r'\b(no|nope|not really)\b'],
                'responses': [
                    "I understand. Let me know if you change your mind.",
                    "That's okay, we can try something else.",
                    "Fair enough. What would work better?",
                    "I see your point. What do you think instead?",
                    "No problem. We can go a different direction.",
                ]
            },
            'compliment': {
                'keywords': ['thank you', 'thanks', 'appreciate', 'great job', 'well done', 'amazing'],
                'patterns': [r'\b(thank|thanks|appreciate)\b'],
                'responses': [
                    "You're very welcome! Happy to help!",
                    "My pleasure! Glad I could assist.",
                    "Thank you for saying that! It means a lot.",
                    "You're too kind! Thanks for the feedback.",
                    "I appreciate that! Thanks for letting me know.",
                ]
            },
            'confusion': {
                'keywords': ['confused', 'don\'t understand', 'what do you mean', 'unclear', 'huh'],
                'patterns': [r'\b(confused|understand|unclear|huh)\b'],
                'responses': [
                    "Let me try to clarify that for you.",
                    "I can see how that might be confusing. Let me explain.",
                    "Good point - let me be more clear about that.",
                    "I should have been clearer. Let me try again.",
                    "That's a fair question. Let me break it down.",
                ]
            },
            'time': {
                'keywords': ['what time', 'time is it', 'current time'],
                'patterns': [r'\b(time|clock)\b'],
                'responses': [
                    "The current time is {time}.",
                    "It's {time} right now.",
                    "According to my clock, it's {time}.",
                    "The time is {time}.",
                ]
            },
            'generic': {
                'keywords': [],
                'patterns': [],
                'responses': [
                    "That's interesting! Tell me more.",
                    "I see what you mean.",
                    "That makes sense.",
                    "I understand what you're saying.",
                    "That's a good point.",
                    "I hear you.",
                    "That sounds reasonable.",
                    "I can relate to that.",
                    "That's worth considering.",
                    "I appreciate you sharing that.",
                ]
            }
        }
    
    def add_custom_template(self, category: str, keywords: List[str], patterns: List[str], responses: List[str]):
        """Add a custom response template category."""
        self.template_responses[category] = {
            'keywords': keywords,
            'patterns': patterns,
            'responses': responses
        }
        logger.info(f"Added custom template category: {category}")
    
    def set_response_mode(self, mode: str):
        """Change the response generation mode."""
        valid_modes = ['echo', 'template']
        if mode not in valid_modes:
            raise ValueError(f"Invalid response mode: {mode}. Valid modes: {valid_modes}")
        
        self.response_mode = mode
        logger.info(f"Response mode changed to: {mode}")
    
    def get_conversation_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get conversation history."""
        if limit:
            return self.conversation_history[-limit:]
        return self.conversation_history.copy()
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        if not self.conversation_history:
            return {'total_exchanges': 0}
        
        total_exchanges = len(self.conversation_history)
        avg_input_length = sum(len(h['input']) for h in self.conversation_history) / total_exchanges
        
        return {
            'total_exchanges': total_exchanges,
            'avg_input_length': round(avg_input_length, 1),
            'response_mode': self.response_mode,
            'template_categories': list(self.template_responses.keys()),
        }