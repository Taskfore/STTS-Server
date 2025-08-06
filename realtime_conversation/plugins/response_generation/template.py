"""
Template-based response generator.

Generates responses using predefined templates with keyword matching
and variable substitution.
"""

import logging
import random
import re
from typing import Dict, Any, List
from datetime import datetime
from .base import BaseResponseGenerator
from ...core.interfaces import TranscriptionResult, ConversationContext

logger = logging.getLogger(__name__)


class TemplateResponseGenerator(BaseResponseGenerator):
    """Template-based response generator with keyword matching."""
    
    def __init__(self, template_file: str = None):
        """
        Initialize template response generator.
        
        Args:
            template_file: Path to template configuration file
        """
        super().__init__(response_mode="template")
        self.templates = self._load_default_templates()
        
        if template_file:
            self._load_templates_from_file(template_file)
        
        logger.info(f"Template response generator initialized with {len(self.templates)} categories")
    
    def _generate_response_sync(
        self, 
        input_text: str, 
        transcription: TranscriptionResult,
        context: ConversationContext
    ) -> str:
        """Generate template-based response."""
        try:
            input_lower = input_text.lower()
            keywords = self._extract_keywords(input_text)
            
            # Find matching template category
            matched_category = self._find_best_category(input_lower, keywords)
            
            if matched_category and matched_category in self.templates:
                # Select random response from category
                category_data = self.templates[matched_category]
                responses = category_data.get("responses", [])
                
                if responses:
                    response_template = random.choice(responses)
                    
                    # Apply variable substitution
                    response = self._substitute_variables(
                        response_template, 
                        input_text, 
                        transcription, 
                        context
                    )
                    
                    logger.debug(f"Template response from category '{matched_category}': '{response[:50]}...'")
                    return response
            
            # No matching template, use fallback
            return self._generate_contextual_fallback(input_text, keywords)
            
        except Exception as e:
            logger.error(f"Error in template response generation: {e}")
            return self._get_fallback_response()
    
    def _load_default_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load default response templates."""
        return {
            "greeting": {
                "keywords": ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"],
                "patterns": [r"\b(hello|hi|hey)\b", r"good (morning|afternoon|evening)"],
                "responses": [
                    "Hello! How can I help you today?",
                    "Hi there! What would you like to talk about?",
                    "Hey! Nice to hear from you.",
                    "Good to see you! What's on your mind?",
                    "Hello! I'm here to chat with you."
                ]
            },
            
            "farewell": {
                "keywords": ["goodbye", "bye", "see you", "farewell", "talk later", "catch you later"],
                "patterns": [r"\b(goodbye|bye|farewell)\b", r"see you", r"talk (to you )?later"],
                "responses": [
                    "Goodbye! Have a great day!",
                    "See you later! Take care!",
                    "Bye! It was nice talking with you.",
                    "Farewell! Hope to chat again soon.",
                    "Take care! Until next time!"
                ]
            },
            
            "question_what": {
                "keywords": ["what", "what is", "what are", "what do", "what does"],
                "patterns": [r"\bwhat\s+(is|are|do|does|did|will|would)\b"],
                "responses": [
                    "That's an interesting question about {topic}. Let me think about that.",
                    "You're asking about {topic}. That's a good point to discuss.",
                    "Regarding {topic}, that's something worth exploring.",
                    "I hear you asking about {topic}. What specifically interests you about that?",
                    "That question about {topic} is quite thought-provoking."
                ]
            },
            
            "question_how": {
                "keywords": ["how", "how do", "how can", "how does", "how to"],
                "patterns": [r"\bhow\s+(do|can|does|did|will|would|to)\b"],
                "responses": [
                    "That's a good question about how to {action}.",
                    "You're wondering about the process of {action}. That's practical thinking.",
                    "I understand you want to know how to {action}. That's useful to learn.",
                    "The question of how to {action} is quite common. What aspect interests you most?",
                    "Figuring out how to {action} is important. What's your experience with this so far?"
                ]
            },
            
            "question_why": {
                "keywords": ["why", "why do", "why does", "why is", "why are"],
                "patterns": [r"\bwhy\s+(do|does|is|are|did|would|will)\b"],
                "responses": [
                    "That's a thoughtful question about why {topic}.",
                    "You're curious about the reasons behind {topic}. That shows good critical thinking.",
                    "The question of why {topic} is quite philosophical.",
                    "I can see you're thinking deeply about why {topic}.",
                    "That's an insightful question about the nature of {topic}."
                ]
            },
            
            "affirmation": {
                "keywords": ["yes", "yeah", "sure", "okay", "alright", "absolutely", "definitely"],
                "patterns": [r"\b(yes|yeah|sure|okay|alright|absolutely|definitely)\b"],
                "responses": [
                    "Great! I'm glad you agree.",
                    "Excellent! That's a positive response.",
                    "Perfect! We're on the same page.",
                    "Wonderful! I appreciate your enthusiasm.",
                    "Awesome! It's good to see we understand each other."
                ]
            },
            
            "negation": {
                "keywords": ["no", "nope", "not really", "i don't think", "disagree"],
                "patterns": [r"\b(no|nope)\b", r"not really", r"don't think", r"disagree"],
                "responses": [
                    "I understand you have a different perspective on this.",
                    "That's okay, everyone has their own opinion.",
                    "I see you're not convinced. What are your thoughts?",
                    "Fair enough. What would you prefer instead?",
                    "I respect your viewpoint. Can you tell me more about why you feel that way?"
                ]
            },
            
            "emotion_positive": {
                "keywords": ["happy", "excited", "great", "awesome", "wonderful", "fantastic", "love", "enjoy"],
                "patterns": [r"\b(happy|excited|great|awesome|wonderful|fantastic)\b", r"\b(love|enjoy)\b"],
                "responses": [
                    "That's wonderful to hear! Your positivity is contagious.",
                    "I'm so glad you're feeling {emotion}! That's great news.",
                    "It makes me happy to know you're {emotion} about this.",
                    "Your enthusiasm about {topic} is really inspiring!",
                    "That's fantastic! Positive energy like yours is amazing."
                ]
            },
            
            "emotion_negative": {
                "keywords": ["sad", "upset", "frustrated", "angry", "disappointed", "worried", "stressed"],
                "patterns": [r"\b(sad|upset|frustrated|angry|disappointed|worried|stressed)\b"],
                "responses": [
                    "I'm sorry to hear you're feeling {emotion}. That must be difficult.",
                    "It sounds like you're going through a tough time with {topic}.",
                    "I understand that {topic} can be {emotion}. Your feelings are valid.",
                    "Thank you for sharing that you're feeling {emotion}. Sometimes talking helps.",
                    "I hear that you're {emotion} about {topic}. What would help you feel better?"
                ]
            },
            
            "weather": {
                "keywords": ["weather", "sunny", "rainy", "cloudy", "hot", "cold", "temperature", "forecast"],
                "patterns": [r"\b(weather|sunny|rainy|cloudy|hot|cold|temperature|forecast)\b"],
                "responses": [
                    "Weather is always an interesting topic! How's the weather where you are?",
                    "I hope the weather is nice wherever you are today.",
                    "Weather certainly affects our mood and activities, doesn't it?",
                    "Talking about weather is a great way to start a conversation!",
                    "The weather can really influence how we feel about the day."
                ]
            },
            
            "time": {
                "keywords": ["time", "clock", "morning", "afternoon", "evening", "night", "today", "tomorrow"],
                "patterns": [r"\b(time|clock|morning|afternoon|evening|night|today|tomorrow)\b"],
                "responses": [
                    "Time is such an interesting concept, isn't it?",
                    "I hope you're having a good {time_period}!",
                    "Time seems to pass differently depending on what we're doing.",
                    "It's nice to take a moment and be present in time.",
                    "Whether it's morning, afternoon, or evening, each time has its own character."
                ]
            },
            
            "help": {
                "keywords": ["help", "assist", "support", "guide", "explain", "teach", "show me"],
                "patterns": [r"\b(help|assist|support|guide|explain|teach)\b", r"show me"],
                "responses": [
                    "I'd be happy to help you with {topic}! What specifically do you need?",
                    "Of course! I'm here to assist you. What can I help you understand?",
                    "I'll do my best to support you with {topic}. What's your main concern?",
                    "Absolutely! Helping you learn about {topic} sounds great.",
                    "I'm glad you asked for help with {topic}. Let's work through this together."
                ]
            }
        }
    
    def _load_templates_from_file(self, template_file: str) -> None:
        """Load templates from a configuration file."""
        try:
            import yaml
            from pathlib import Path
            
            file_path = Path(template_file)
            if not file_path.exists():
                logger.warning(f"Template file not found: {template_file}")
                return
            
            with open(file_path, 'r', encoding='utf-8') as f:
                custom_templates = yaml.safe_load(f) or {}
            
            # Merge with default templates
            self.templates.update(custom_templates)
            logger.info(f"Loaded templates from file: {template_file}")
            
        except Exception as e:
            logger.error(f"Error loading template file {template_file}: {e}")
    
    def _find_best_category(self, input_lower: str, keywords: List[str]) -> str:
        """Find the best matching template category."""
        best_category = None
        best_score = 0
        
        for category, data in self.templates.items():
            score = self._calculate_match_score(input_lower, keywords, data)
            
            if score > best_score:
                best_score = score
                best_category = category
        
        # Require minimum score for match
        if best_score < 0.3:
            return None
        
        logger.debug(f"Best category match: '{best_category}' (score: {best_score:.2f})")
        return best_category
    
    def _calculate_match_score(self, input_lower: str, keywords: List[str], template_data: Dict[str, Any]) -> float:
        """Calculate match score for a template category."""
        score = 0.0
        
        # Check keyword matches
        template_keywords = template_data.get("keywords", [])
        for keyword in template_keywords:
            if keyword in input_lower:
                score += 1.0
        
        # Check pattern matches
        template_patterns = template_data.get("patterns", [])
        for pattern in template_patterns:
            try:
                if re.search(pattern, input_lower):
                    score += 1.5  # Patterns get higher weight
            except re.error:
                continue
        
        # Normalize by number of criteria
        total_criteria = len(template_keywords) + len(template_patterns)
        if total_criteria > 0:
            score = score / total_criteria
        
        return score
    
    def _substitute_variables(
        self, 
        template: str, 
        input_text: str, 
        transcription: TranscriptionResult,
        context: ConversationContext
    ) -> str:
        """Substitute variables in response template."""
        try:
            result = template
            
            # Extract potential topics (noun phrases)
            topics = self._extract_topics(input_text)
            main_topic = topics[0] if topics else "that"
            
            # Extract potential actions (verb phrases)
            actions = self._extract_actions(input_text)
            main_action = actions[0] if actions else "do that"
            
            # Extract emotion words
            emotions = self._extract_emotions(input_text)
            main_emotion = emotions[0] if emotions else "concerned"
            
            # Extract time references
            time_refs = self._extract_time_references(input_text)
            time_period = time_refs[0] if time_refs else "time"
            
            # Perform substitutions
            substitutions = {
                "{topic}": main_topic,
                "{action}": main_action,
                "{emotion}": main_emotion,
                "{time_period}": time_period,
                "{input}": input_text,
                "{language}": transcription.language,
                "{user_name}": context.user_data.get("name", "friend") if context.user_data else "friend"
            }
            
            for placeholder, value in substitutions.items():
                result = result.replace(placeholder, value)
            
            return result
            
        except Exception as e:
            logger.error(f"Error substituting variables: {e}")
            return template  # Return original template on error
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract potential topic nouns from text."""
        import re
        
        # Simple noun extraction (words that could be topics)
        # This is a basic implementation - could be improved with NLP
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Common topic indicators
        topic_words = []
        for word in words:
            if len(word) > 3 and word not in {
                "this", "that", "they", "them", "what", "where", "when", "how", "why",
                "have", "been", "will", "would", "could", "should", "might"
            }:
                topic_words.append(word)
        
        return topic_words[:3]  # Return up to 3 topics
    
    def _extract_actions(self, text: str) -> List[str]:
        """Extract potential action verbs from text."""
        import re
        
        # Look for verb patterns
        verb_patterns = [
            r'\b(do|does|doing|done)\s+(\w+)',
            r'\b(make|makes|making|made)\s+(\w+)',
            r'\b(get|gets|getting|got)\s+(\w+)',
            r'\b(go|goes|going|went)\s+(\w+)',
            r'\bhow\s+to\s+(\w+)',
            r'\bwant\s+to\s+(\w+)',
            r'\bneed\s+to\s+(\w+)'
        ]
        
        actions = []
        text_lower = text.lower()
        
        for pattern in verb_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if isinstance(match, tuple):
                    actions.extend([m for m in match if len(m) > 2])
                elif len(match) > 2:
                    actions.append(match)
        
        return actions[:3]  # Return up to 3 actions
    
    def _extract_emotions(self, text: str) -> List[str]:
        """Extract emotion words from text."""
        emotion_words = [
            "happy", "sad", "excited", "angry", "frustrated", "worried", "stressed",
            "anxious", "calm", "peaceful", "nervous", "confident", "proud", "ashamed",
            "grateful", "disappointed", "surprised", "confused", "curious", "bored"
        ]
        
        text_lower = text.lower()
        found_emotions = [emotion for emotion in emotion_words if emotion in text_lower]
        
        return found_emotions
    
    def _extract_time_references(self, text: str) -> List[str]:
        """Extract time references from text."""
        time_words = [
            "morning", "afternoon", "evening", "night", "today", "tomorrow", 
            "yesterday", "now", "later", "soon", "early", "late"
        ]
        
        text_lower = text.lower()
        found_times = [time_word for time_word in time_words if time_word in text_lower]
        
        return found_times
    
    def _generate_contextual_fallback(self, input_text: str, keywords: List[str]) -> str:
        """Generate contextual fallback response when no template matches."""
        if keywords:
            # Use keywords to create a more contextual response
            main_keyword = keywords[0]
            return f"That's interesting that you mentioned {main_keyword}. Can you tell me more about that?"
        else:
            # Generic fallback responses
            fallbacks = [
                "That's an interesting point. What made you think of that?",
                "I'd like to hear more about your thoughts on this.",
                "That's worth discussing. What's your perspective?",
                "I find that fascinating. Can you elaborate?",
                "That's a good observation. What else comes to mind?"
            ]
            return random.choice(fallbacks)
    
    # Configuration methods
    
    def add_template_category(self, category: str, template_data: Dict[str, Any]) -> None:
        """Add a new template category."""
        self.templates[category] = template_data
        logger.info(f"Added template category: {category}")
    
    def get_template_categories(self) -> List[str]:
        """Get list of available template categories."""
        return list(self.templates.keys())
    
    def get_config(self) -> Dict[str, Any]:
        """Get template generator configuration."""
        config = super().get_config()
        config.update({
            "template_categories": len(self.templates),
            "available_categories": self.get_template_categories()
        })
        return config


# Factory functions

def create_template_generator(template_file: str = None) -> TemplateResponseGenerator:
    """
    Create a template-based response generator.
    
    Args:
        template_file: Optional path to custom template file
        
    Returns:
        Configured template response generator
    """
    return TemplateResponseGenerator(template_file=template_file)


def create_template_generator_from_config(config: Dict[str, Any]) -> TemplateResponseGenerator:
    """
    Create template response generator from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured template response generator
    """
    return TemplateResponseGenerator(
        template_file=config.get("template_file")
    )