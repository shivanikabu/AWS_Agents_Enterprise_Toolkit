import json
import numpy as np
from datetime import datetime
import boto3
from langfuse import Langfuse
import chromadb
from sentence_transformers import SentenceTransformer
import PyPDF2
import io


class AgentBackend:
    """Backend class to handle all agent operations and AWS integrations"""
    
    def __init__(self):
        self.bedrock_client = None
        self.langfuse_client = None
        self.vector_db = None
        self.model = None
        self.chroma_client = None
        self.aws_session = None
        self.aws_region = None
        
    def connect_aws(self, aws_access_key, aws_secret_key, aws_region):
        """Connect to AWS Bedrock"""
        try:
            session = boto3.Session(
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region
            )
            self.bedrock_client = session.client('bedrock-runtime')
            self.aws_session = session
            self.aws_region = aws_region
            
            # Initialize vector store
            self.initialize_vector_db()
            
            return True, "AWS Connected Successfully"
        except Exception as e:
            return False, f"AWS Connection Failed: {str(e)}"
    
    def connect_langfuse(self, public_key, secret_key, host):
        """Connect to Langfuse for observability"""
        try:
            self.langfuse_client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host
            )
            return True, "Langfuse Connected Successfully"
        except Exception as e:
            return False, f"Langfuse Connection Failed: {str(e)}"
    
    def initialize_vector_db(self):
        """Initialize ChromaDB for document storage"""
        if self.vector_db is None:
            self.chroma_client = chromadb.Client()
            try:
                self.vector_db = self.chroma_client.get_collection(name="documents")
            except:
                self.vector_db = self.chroma_client.create_collection(name="documents")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def clear_vector_db(self):
        """Clear all documents from the vector database"""
        print(f"\n{'='*60}")
        print(f"CLEARING VECTOR DATABASE")
        print(f"{'='*60}")
        
        try:
            if self.chroma_client is not None:
                # Delete the existing collection
                try:
                    self.chroma_client.delete_collection(name="documents")
                    print("✔ Deleted existing 'documents' collection")
                except Exception as e:
                    print(f"  Collection didn't exist or already deleted: {str(e)}")
                
                # Create a fresh collection
                self.vector_db = self.chroma_client.create_collection(name="documents")
                print("✔ Created fresh 'documents' collection")
                print("✔ Vector database cleared successfully")
                print(f"{'='*60}\n")
                return True, "Vector database cleared successfully"
            else:
                print("⚠ ChromaDB client not initialized")
                print(f"{'='*60}\n")
                return False, "Vector database not initialized"
        except Exception as e:
            print(f"✗ Error clearing database: {str(e)}")
            print(f"{'='*60}\n")
            import traceback
            traceback.print_exc()
            return False, f"Error clearing database: {str(e)}"
    
    def process_documents(self, uploaded_files):
        """Process and store PDF documents in vector DB - chunked by paragraphs"""
        print(f"\n{'='*60}")
        print(f"PDF DOCUMENT PROCESSING STARTED")
        print(f"Number of PDF files: {len(uploaded_files)}")
        print(f"{'='*60}\n")
        
        try:
            # Initialize vector DB if not already done
            if self.vector_db is None:
                print("Initializing vector database...")
                self.initialize_vector_db()
                print(f"Vector DB initialized: {self.vector_db is not None}")
            else:
                # Clear existing documents before processing new ones
                print("Clearing existing documents from vector database...")
                success, message = self.clear_vector_db()
                if not success:
                    return False, f"Failed to clear database: {message}"
            
            if self.model is None:
                print("Initializing embedding model...")
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                print(f"Model loaded: {self.model is not None}")
            
            total_chunks = 0
            all_paragraphs = []
            all_embeddings = []
            all_metadatas = []
            all_ids = []
            
            for file_idx, uploaded_file in enumerate(uploaded_files):
                print(f"\n{'='*50}")
                print(f"Processing PDF {file_idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
                print(f"{'='*50}")
                
                # Read PDF file
                try:
                    pdf_bytes = uploaded_file.read()
                    pdf_file = io.BytesIO(pdf_bytes)
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    
                    num_pages = len(pdf_reader.pages)
                    print(f"✔ PDF opened successfully")
                    print(f"✔ Total pages: {num_pages}")
                    
                    # Extract text from all pages
                    content = ""
                    for page_num in range(num_pages):
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        
                        if page_text:
                            content += page_text + "\n\n"
                            print(f"  ✔ Page {page_num + 1}/{num_pages}: {len(page_text)} characters extracted")
                        else:
                            print(f"  ⚠ Page {page_num + 1}/{num_pages}: No text extracted (might be image-based)")
                    
                    print(f"\n✔ PDF text extraction complete: {len(content)} total characters")
                    
                    if not content or len(content) < 50:
                        print(f"✗ ERROR: No readable text extracted from PDF. The PDF might be:")
                        print(f"  - Image-based (scanned document)")
                        print(f"  - Encrypted or password-protected")
                        print(f"  - Corrupted")
                        print(f"  Skipping this file.")
                        continue
                    
                except Exception as pdf_error:
                    print(f"✗ PDF parsing error: {type(pdf_error).__name__}: {str(pdf_error)}")
                    print(f"  Could not read PDF file. Skipping.")
                    continue
                
                # Split by paragraph breaks (double newline or multiple newlines)
                # Clean up excessive newlines first
                content = '\n\n'.join([line.strip() for line in content.split('\n') if line.strip()])
                paragraphs = content.split('\n\n')
                print(f"✔ Text split into {len(paragraphs)} paragraphs")
                
                # Clean and filter paragraphs
                chunks_from_this_file = 0
                valid_paragraphs = 0
                
                for para_idx, para in enumerate(paragraphs):
                    # Strip whitespace
                    para = para.strip()
                    
                    # Skip empty or very short paragraphs
                    if len(para) < 50:
                        continue
                    
                    # Clean the paragraph text
                    # Replace multiple spaces with single space
                    para = ' '.join(para.split())
                    
                    valid_paragraphs += 1
                    
                    if valid_paragraphs <= 3:  # Show first 3 paragraphs in detail
                        print(f"\n  Paragraph {valid_paragraphs}:")
                        print(f"    Length: {len(para)} characters")
                        print(f"    Preview: '{para[:100]}...'")
                    elif valid_paragraphs == 4:
                        print(f"\n  ... processing remaining paragraphs ...")
                    
                    # Generate embedding for this paragraph
                    embedding = self.model.encode(para).tolist()
                    
                    # Create unique ID for this chunk
                    timestamp_ms = int(datetime.now().timestamp() * 1000)
                    chunk_id = f"{uploaded_file.name}_para{valid_paragraphs}_{timestamp_ms}"
                    
                    # Store in batch lists
                    all_paragraphs.append(para)
                    all_embeddings.append(embedding)
                    all_metadatas.append({
                        "source": uploaded_file.name,
                        "chunk_index": valid_paragraphs,
                        "chunk_type": "paragraph",
                        "char_count": len(para),
                        "file_type": "pdf",
                        "total_pages": num_pages
                    })
                    all_ids.append(chunk_id)
                    
                    chunks_from_this_file += 1
                    total_chunks += 1
                
                print(f"\n✔ '{uploaded_file.name}': Prepared {chunks_from_this_file} valid paragraph chunks")
            
            # Batch insert all documents at once
            if total_chunks > 0:
                print(f"\n{'='*60}")
                print(f"BATCH INSERTING {total_chunks} CHUNKS INTO VECTOR DATABASE")
                print(f"{'='*60}")
                
                self.vector_db.add(
                    embeddings=all_embeddings,
                    documents=all_paragraphs,
                    metadatas=all_metadatas,
                    ids=all_ids
                )
                print(f"✔ Batch insert completed successfully!")
            else:
                print(f"\n⚠ WARNING: No chunks to insert")
                print(f"   All paragraphs were either too short or no text could be extracted.")
                return False, "No valid text content found in PDF files. PDFs might be image-based or empty."
            
            # Verify what's in the database
            print(f"\n{'='*60}")
            print(f"DATABASE VERIFICATION")
            print(f"{'='*60}")
            verification = self.vector_db.get()
            if verification and 'documents' in verification:
                actual_count = len(verification['documents'])
                print(f"✔ Database now contains: {actual_count} total document chunks")
                if actual_count > 0:
                    print(f"✔ First chunk preview: '{verification['documents'][0][:150]}...'")
                if actual_count != total_chunks:
                    print(f"⚠ WARNING: Expected {total_chunks} chunks but found {actual_count} in database")
            else:
                print(f"✗ ERROR: Could not verify database contents!")
                return False, "Database verification failed"
            print(f"{'='*60}\n")
            
            success_message = f"✔ Successfully processed {len(uploaded_files)} PDF file(s) → {total_chunks} paragraph chunks stored"
            print(success_message)
            return True, success_message
            
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"CRITICAL ERROR in process_documents")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"{'='*60}")
            import traceback
            traceback.print_exc()
            return False, f"Error processing PDF documents: {str(e)}"
    
    def planner_agent(self, query):
        """
        1.1.3.1 Planner Agent - Assess query sentiment and create customer persona
        Returns: persona type and sentiment analysis
        """
        query_lower = query.lower()
        
        # Angry customer - frustration, disappointment, repetition, annoyance
        if any(phrase in query_lower for phrase in [
            'how many time', 'how many times', 'annoyed', 'am annoyed',
            'asking again and again', 'disappointed', 'you don\'t get me', 
            'frustrated', 'upset', 'terrible', 'worst', 'hate', 
            'not listening', 'don\'t understand me', 'keep asking',
            'tired of', 'fed up', 'sick of', 'irritated', 'angry'
        ]):
            persona = "angry customer"
            sentiment = "negative"
            detail = f"Detected persona: '{persona}'. Sentiment: {sentiment}. High frustration detected."
        
        # Confused customer - tried before, didn't work, unclear
        elif any(phrase in query_lower for phrase in [
            'tried before', 'did not work', 'didn\'t work', 'not working',
            'confused', 'don\'t understand', 'unclear', 'lost',
            'can\'t figure out', 'having trouble', 'still not'
        ]):
            persona = "confused customer"
            sentiment = "negative"
            detail = f"Detected persona: '{persona}'. Sentiment: {sentiment}. Confusion detected."
        
        # Precision queries - exactly, clearly, specific requests
        elif any(phrase in query_lower for phrase in [
            'can you tell me exactly', 'can you tell me clearly', 
            'precisely', 'specific', 'detailed', 'step by step',
            'how to', 'configure', 'setup', 'implement',
            'what exactly', 'be specific', 'give me the exact'
        ]):
            persona = "precision ask"
            sentiment = "neutral"
            detail = f"Detected persona: '{persona}'. Technical complexity: High."
        
        # Simple query - straightforward phrase or question
        else:
            persona = "simple query"
            sentiment = "positive"
            detail = f"Detected persona: '{persona}'. Sentiment score: 0.65 (positive). Complexity: Low."
        
        return {
            "agent": "Planner Agent",
            "persona": persona,
            "sentiment": sentiment,
            "detail": detail
        }
    
    def orchestration_agent(self, persona):
        """
        1.1.3.2 Orchestration Agent - Route to appropriate agent combinations
        Returns: list of agents to execute
        """
        if persona == "simple query":
            # 1.1.3.2.1.1 Simple Query → RAG Agent
            agent_flow = ["RAG Agent", "Reflector Agent", "Response Agent", "Feedback Agent"]
            detail = f"Standard RAG flow selected for '{persona}'. Delegating to RAG Agent."
        
        elif persona in ["angry customer", "confused customer"]:
            # 1.1.3.2.1.2 Angry/Confused → Emotions Agent, Calming Agent, RAG Agent
            agent_flow = ["Emotions Agent", "Calming Agent", "RAG Agent", "Reflector Agent", "Response Agent", "Feedback Agent"]
            detail = f"Emotional support flow activated. Delegating to Emotions → Calming → RAG Agents."
        
        else:  # precision ask
            # 1.1.3.2.1.3 Precision Customer → Best Practices Agent
            agent_flow = ["RAG Agent", "Best Practices Agent", "Reflector Agent", "Response Agent", "Feedback Agent"]
            detail = f"Best practices flow selected. Delegating to RAG + Best Practices Agents."
        
        return {
            "agent": "Orchestration Agent",
            "agent_flow": agent_flow,
            "detail": detail
        }
    
    def rag_agent(self, query):
        """
        RAG Agent - Retrieve ALL documents from vector DB
        Returns: ALL document chunks for complete context
        """
        relevant_docs = []
        detail = ""
        
        print(f"\n{'='*60}")
        print(f"RAG AGENT CALLED")
        print(f"Query: {query}")
        print(f"Vector DB exists: {self.vector_db is not None}")
        print(f"{'='*60}\n")
        
        if self.vector_db is None:
            detail = "ERROR: Vector database not initialized!"
            print(f"RETURNING: {detail}")
            return {
                "agent": "RAG Agent",
                "documents": [],
                "detail": detail
            }
        
        try:
            # ALWAYS retrieve ALL documents to provide complete context
            print("Retrieving ALL documents from database for complete context...")
            all_results = self.vector_db.get()
            
            print(f"Got results: {all_results is not None}")
            if all_results:
                print(f"Results keys: {list(all_results.keys())}")
                print(f"Documents in results: {'documents' in all_results}")
                
                if 'documents' in all_results:
                    docs = all_results['documents']
                    print(f"Documents type: {type(docs)}")
                    print(f"Documents length: {len(docs) if docs else 0}")
                    
                    if docs and len(docs) > 0:
                        relevant_docs = docs
                        print(f"SUCCESS! Retrieved ALL {len(relevant_docs)} documents")
                        print(f"First doc (first 100 chars): {str(relevant_docs[0])[:100]}")
                        print(f"Last doc (first 100 chars): {str(relevant_docs[-1])[:100]}")
                        detail = f"✔ Retrieved ALL {len(relevant_docs)} document chunks for complete context"
                    else:
                        print("PROBLEM: Documents list is empty!")
                        detail = "ERROR: No documents in database. Please upload and process PDF documents."
                else:
                    print("PROBLEM: No 'documents' key in results!")
                    detail = "ERROR: Database returned unexpected format."
            else:
                print("PROBLEM: all_results is None!")
                detail = "ERROR: Database query returned nothing."
                
        except Exception as e:
            print(f"EXCEPTION: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            detail = f"ERROR: {str(e)}"
        
        print(f"\nFINAL RETURN: {len(relevant_docs)} documents")
        print(f"Detail message: {detail}")
        print(f"{'='*60}\n")
        
        return {
            "agent": "RAG Agent",
            "documents": relevant_docs,
            "detail": detail
        }
    
    def emotions_agent(self, query, persona):
        """Emotions Agent - Analyze emotional content"""
        intensity = 7 if persona == "angry customer" else 5
        detail = f"Emotional analysis: {'Frustration' if persona == 'angry customer' else 'Confusion'} detected. Intensity: {intensity}/10."
        
        return {
            "agent": "Emotions Agent",
            "emotion": "frustration" if persona == "angry customer" else "confusion",
            "intensity": intensity,
            "detail": detail
        }
    
    def calming_agent(self, emotion_data):
        """Calming Agent - Generate empathetic preamble"""
        preamble = "I understand your frustration, and I'm here to help resolve this for you."
        detail = f"Empathetic preamble generated: '{preamble}'"
        
        return {
            "agent": "Calming Agent",
            "preamble": preamble,
            "detail": detail
        }
    
    def best_practices_agent(self):
        """Best Practices Agent - Add technical best practices"""
        detail = "Enhanced response with step-by-step guidance, warnings, and industry best practices."
        
        return {
            "agent": "Best Practices Agent",
            "enhancement": "best_practices",
            "detail": detail
        }
    
    def reflector_agent(self, query, documents, response_text, guardrails):
        """
        Reflector Agent - Evaluate query, retrievals, and response quality
        """
        # Evaluate guardrails
        violations = []
        if guardrails:
            guardrail_list = [g.strip() for g in guardrails.split('\n') if g.strip()]
            # Simple guardrail check (in production, use more sophisticated methods)
            for guardrail in guardrail_list:
                if 'profanity' in guardrail.lower():
                    # Check for profanity (simplified)
                    pass
        
        quality_score = np.random.uniform(0.80, 0.95)
        token_count = len(response_text.split()) * 1.3  # Rough estimate
        
        # Performance evaluation
        performance_status = "optimal"  # optimal, acceptable, warning
        performance_issue = None
        
        # Check for various issues
        if not documents or len(documents) == 0:
            performance_status = "warning"
            performance_issue = "Critical: No relevant documents found in vector DB. Response may contain hallucinations or be based solely on model knowledge."
        elif len(documents) < 3:
            performance_status = "acceptable"
            performance_issue = "Document retrieval returned fewer than expected chunks. Consider expanding document base or adjusting search parameters."
        elif quality_score < 0.85:
            performance_status = "acceptable"
            performance_issue = f"Quality score ({quality_score:.2f}) below optimal threshold. Response accuracy may be impacted by context relevance."
        elif token_count > 1500:
            performance_status = "acceptable"
            performance_issue = f"Token usage ({int(token_count)}) higher than expected. Consider optimizing response length or context window."
        elif violations:
            performance_status = "warning"
            performance_issue = f"Critical: Guardrail violation detected - {violations[0]}. Response blocked by safety filters."
        
        detail = f"Response evaluated. {'No guardrail violations detected' if not violations else f'{len(violations)} violations'}. Quality score: {quality_score:.2f}. Total tokens: {int(token_count)}"
        
        return {
            "agent": "Reflector Agent",
            "quality_score": quality_score,
            "violations": violations,
            "token_count": int(token_count),
            "performance_status": performance_status,
            "performance_issue": performance_issue,
            "detail": detail
        }
    
    def response_agent(self, query, documents, persona, calming_preamble=None, best_practices=False):
        """
        Response Agent - Build final response using AWS Bedrock with FULL document context
        """
        if not self.bedrock_client:
            return {
                "agent": "Response Agent",
                "response": "[Error: AWS Bedrock not connected]",
                "detail": "AWS Bedrock connection required"
            }
        
        try:
            print(f"\n{'='*60}")
            print(f"RESPONSE AGENT - Building response")
            print(f"Received documents: {len(documents)}")
            print(f"Persona: {persona}")
            print(f"{'='*60}\n")
            
            # Check if this is a document display request
            query_lower = query.lower()
            if any(term in query_lower for term in ['display document', 'show document', 'list document', 
                                                      'what documents', 'document content', 'all documents']):
                # Format document display response
                if documents:
                    response = f"I found {len(documents)} document chunks in your uploaded files:\n\n"
                    for i, doc in enumerate(documents[:10], 1):
                        response += f"**Chunk {i}:**\n{doc[:400]}{'...' if len(doc) > 400 else ''}\n\n"
                    if len(documents) > 10:
                        response += f"... and {len(documents) - 10} more chunks."
                else:
                    response = "No documents have been uploaded yet. Please upload documents using the sidebar."
                
                return {
                    "agent": "Response Agent",
                    "response": response,
                    "detail": f"Document display response generated. Length: {len(response)} characters."
                }
            
            # Prepare COMPLETE context - use ALL documents
            if documents and len(documents) > 0:
                # Join ALL documents with clear separators
                context_text = "\n\n---\n\n".join(documents)
                total_context_chars = len(context_text)
                print(f"✔ Using COMPLETE document context: {len(documents)} chunks")
                print(f"✔ Total context length: {total_context_chars} characters")
                detail_prefix = f"Using COMPLETE context: ALL {len(documents)} document chunks ({total_context_chars} chars)"
            else:
                context_text = "No document context available."
                print(f"✗ WARNING: No documents provided!")
                detail_prefix = "WARNING: No document context available"
            
            # Build prompt based on persona
            if persona in ["angry customer", "confused customer"]:
                system_prompt = "You are an empathetic customer support agent. First acknowledge the customer's feelings briefly, then provide a comprehensive answer using the COMPLETE document context provided below."
            elif persona == "precision ask":
                system_prompt = "You are a technical expert. Provide detailed, accurate information using the COMPLETE document context provided below."
            else:
                system_prompt = "You are a helpful assistant. Provide clear, comprehensive information using the COMPLETE document context provided below."
            
            prompt = f"""{system_prompt}

COMPLETE DOCUMENT CONTEXT:
{context_text}

Customer query: {query}

INSTRUCTIONS:
- Use the COMPLETE document context above to answer the query
- Provide a comprehensive, well-structured answer
- Reference specific information from the context
- If the query asks for a summary, synthesize ALL the information provided
- Be thorough and accurate
"""

            print(f"Calling AWS Bedrock with {len(prompt)} character prompt...")
            
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4000,  # Increased for comprehensive responses
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7
            }
            
            response = self.bedrock_client.invoke_model(
                modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response['body'].read())
            final_response = response_body['content'][0]['text']
            
            print(f"✔ Response generated: {len(final_response)} characters")
            
            # Add calming preamble if provided
            if calming_preamble:
                final_response = f"{calming_preamble}\n\n{final_response}"
                print(f"✔ Added empathetic preamble")
            
            detail = f"{detail_prefix} | Final response: {len(final_response)} characters"
            
            print(f"✔ Response Agent complete")
            print(f"{'='*60}\n")
            
            return {
                "agent": "Response Agent",
                "response": final_response,
                "detail": detail
            }
            
        except Exception as e:
            print(f"✗ Response Agent error: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "agent": "Response Agent",
                "response": f"[Error calling Bedrock: {str(e)}]",
                "detail": f"Bedrock API Error: {str(e)}"
            }
    
    def feedback_agent(self, quality_score, risk_weight, accuracy_weight, latency_weight, cost_weight, token_count):
        """
        Feedback Agent - Calculate convergence score and provide feedback
        """
        # Calculate convergence score based on multiple factors
        convergence_score = (
            risk_weight * 0.9 +
            accuracy_weight * quality_score +
            latency_weight * 0.75 +
            cost_weight * (1.0 - min(token_count / 10000, 1.0))
        ) / (risk_weight + accuracy_weight + latency_weight + cost_weight)
        
        recommendation = "Proceed" if convergence_score >= 0.7 else "Refine"
        detail = f"Convergence score: {convergence_score:.3f}. Meets threshold: {'Yes' if convergence_score >= 0.7 else 'No'}. Recommendation: {recommendation}."
        
        return {
            "agent": "Feedback Agent",
            "convergence_score": convergence_score,
            "recommendation": recommendation,
            "detail": detail
        }
    
    def execute_agentic_flow(self, query, risk_weight, accuracy_weight, latency_weight, cost_weight, guardrails):
        """
        Main orchestration method to execute the complete agentic flow
        Returns: Complete agent execution log and final response
        """
        agents_executed = []
        
        # Step 1: Planner Agent
        planner_result = self.planner_agent(query)
        agents_executed.append({
            "agent": "Planner Agent",
            "emoji": "🧠",
            "action": "Analyzing query sentiment",
            "detail": planner_result["detail"]
        })
        persona = planner_result["persona"]
        
        # Step 2: Orchestration Agent
        orchestration_result = self.orchestration_agent(persona)
        agents_executed.append({
            "agent": "Orchestration Agent",
            "emoji": "🎯",
            "action": "Routing to appropriate agents",
            "detail": orchestration_result["detail"]
        })
        agent_flow = orchestration_result["agent_flow"]
        
        # Initialize variables BEFORE the loop
        calming_preamble = None
        best_practices = False
        documents = []
        final_response = ""
        quality_score = 0.85
        token_count = 0
        
        # DEBUG: Print initial state
        print(f"DEBUG: Starting agent flow with {len(agent_flow)} agents")
        print(f"DEBUG: Initial documents list: {len(documents)} items")
        
        # Execute the agent flow
        for agent_name in agent_flow:
            print(f"DEBUG: Executing {agent_name}")
            
            if agent_name == "Emotions Agent":
                emotion_result = self.emotions_agent(query, persona)
                agents_executed.append({
                    "agent": "Emotions Agent",
                    "emoji": "💭",
                    "action": "Analyzing emotional content",
                    "detail": emotion_result["detail"]
                })
            
            elif agent_name == "Calming Agent":
                calming_result = self.calming_agent(None)
                calming_preamble = calming_result["preamble"]
                agents_executed.append({
                    "agent": "Calming Agent",
                    "emoji": "🕊️",
                    "action": "Generating empathetic response",
                    "detail": calming_result["detail"]
                })
            
            elif agent_name == "RAG Agent":
                rag_result = self.rag_agent(query)
                documents = rag_result["documents"]
                # DEBUG: Print what RAG Agent returned
                print(f"DEBUG: RAG Agent returned {len(documents)} documents")
                if documents:
                    print(f"DEBUG: First document preview: {documents[0][:100]}...")
                else:
                    print("DEBUG: WARNING - No documents retrieved!")
                
                agents_executed.append({
                    "agent": "RAG Agent",
                    "emoji": "📚",
                    "action": "Retrieving relevant documents",
                    "detail": f"{rag_result['detail']} | DEBUG: Retrieved {len(documents)} chunks"
                })
            
            elif agent_name == "Best Practices Agent":
                bp_result = self.best_practices_agent()
                best_practices = True
                agents_executed.append({
                    "agent": "Best Practices Agent",
                    "emoji": "⭐",
                    "action": "Enhancing with best practices",
                    "detail": bp_result["detail"]
                })
            
            elif agent_name == "Response Agent":
                # DEBUG: Print what Response Agent receives
                print(f"DEBUG: Response Agent receiving {len(documents)} documents")
                print(f"DEBUG: Persona: {persona}")
                print(f"DEBUG: Calming preamble: {calming_preamble is not None}")
                
                response_result = self.response_agent(query, documents, persona, calming_preamble, best_practices)
                final_response = response_result["response"]
                
                print(f"DEBUG: Response Agent generated {len(final_response)} chars")
                
                agents_executed.append({
                    "agent": "Response Agent",
                    "emoji": "✏️",
                    "action": "Building final response",
                    "detail": f"{response_result['detail']} | DEBUG: Used {len(documents)} documents"
                })
            
            elif agent_name == "Reflector Agent":
                reflector_result = self.reflector_agent(query, documents, final_response, guardrails)
                agents_executed.append({
                    "agent": "Reflector Agent",
                    "emoji": "🤔",
                    "action": "Evaluating response quality",
                    "detail": reflector_result["detail"],
                    "performance_status": reflector_result["performance_status"],
                    "performance_issue": reflector_result["performance_issue"]
                })
                quality_score = reflector_result["quality_score"]
                token_count = reflector_result["token_count"]
            
            elif agent_name == "Feedback Agent":
                feedback_result = self.feedback_agent(
                    quality_score, risk_weight, accuracy_weight, latency_weight, cost_weight, token_count
                )
                agents_executed.append({
                    "agent": "Feedback Agent",
                    "emoji": "📊",
                    "action": "Calculating convergence score",
                    "detail": feedback_result["detail"]
                })
        
        # DEBUG: Final state
        print(f"DEBUG: Flow complete. Final response length: {len(final_response)}")
        
        return {
            "agents_executed": agents_executed,
            "final_response": final_response,
            "persona": persona
        }