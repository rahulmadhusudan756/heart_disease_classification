'use client';

import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Activity, Menu, X, RotateCcw } from 'lucide-react';
import { Message, PredictionResponse, MODELS } from '@/types';
import { predictSingle, detectCompatibleModels } from '@/lib/api';
import ChatMessage from './ChatMessage';
import FileUpload from './FileUpload';

type Phase = 'initial' | 'phase1_complete' | 'phase2_complete';

const PHASE1_MODEL = 'normal_heart';
const SPECIFIC_MODELS = MODELS.filter(m => m.id !== 'normal_heart');

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadedData, setUploadedData] = useState<Record<string, unknown> | null>(null);
  const [phase, setPhase] = useState<Phase>('initial');
  const [phase1Result, setPhase1Result] = useState<PredictionResponse | null>(null);
  const [suggestedModel, setSuggestedModel] = useState<string | null>(null);
  const [showSidebar, setShowSidebar] = useState(true);
  const [isClient, setIsClient] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messageIdRef = useRef(0);

  const getNextMessageId = () => {
    messageIdRef.current += 1;
    return `msg-${Date.now()}-${messageIdRef.current}`;
  };

  const createWelcomeMessage = (): Message => ({
    id: 'welcome-message',
    type: 'bot',
    content: `Welcome to HeartGuard AI — 2-Phase Heart Health Assessment

**How it works:**

**Phase 1 — Initial Screening**
Upload your baseline heart data (JSON format). The system will analyze your fundamental cardiac health indicators.

**Phase 2 — Targeted Analysis**
If Phase 1 indicates potential concerns, upload additional patient data. The system will automatically detect compatible disease models and run detailed analysis.

Upload your heart health JSON file in the sidebar to begin.`,
    timestamp: new Date(),
  });

  useEffect(() => {
    setIsClient(true);
    setMessages([createWelcomeMessage()]);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const addMessage = (message: Omit<Message, 'id' | 'timestamp'>) => {
    const newMessage: Message = {
      ...message,
      id: getNextMessageId(),
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, newMessage]);
    return newMessage.id;
  };

  const updateMessage = (id: string, updates: Partial<Message>) => {
    setMessages(prev =>
      prev.map(msg => (msg.id === id ? { ...msg, ...updates } : msg))
    );
  };

  const handleFileUpload = async (data: Record<string, unknown>, fileName: string) => {
    setUploadedData(data);
    
    if (phase === 'initial') {
      addMessage({
        type: 'user',
        content: `Uploaded: **${fileName}** for initial screening`,
      });

      const processingId = addMessage({
        type: 'bot',
        content: 'Running Phase 1 analysis...',
        isLoading: true,
      });

      setIsProcessing(true);

      try {
        const result = await predictSingle(PHASE1_MODEL, data);
        setPhase1Result(result);

        const hasRisk = result.prediction === 1 || (result.probability && result.probability > 0.3);
        
        if (hasRisk) {
          setPhase('phase1_complete');
          const suggested = suggestModelBasedOnData(data);
          setSuggestedModel(suggested);
          
          updateMessage(processingId, {
            type: 'bot',
            content: `**Phase 1 Complete — Initial Screening Results**`,
            isLoading: false,
            predictions: [{ modelId: PHASE1_MODEL, result }],
          });

          addMessage({
            type: 'bot',
            content: `**Attention Required**

Initial screening detected potential cardiac health concerns.

**Risk Level**: ${result.probability ? `${(result.probability * 100).toFixed(1)}%` : 'Elevated'}

**Next Step — Phase 2:**
Upload any patient data JSON file. The system will auto-detect compatible disease models and run targeted analysis.

Type "skip" to end assessment without Phase 2.`,
          });
        } else {
          // Normal result - reset for new assessment
          setPhase('initial');
          setPhase1Result(null);
          setSuggestedModel(null);
          setUploadedData(null);
          
          updateMessage(processingId, {
            type: 'bot',
            content: `**Phase 1 Complete — Initial Screening Results**`,
            isLoading: false,
            predictions: [{ modelId: PHASE1_MODEL, result }],
          });

          addMessage({
            type: 'bot',
            content: `**Results: Normal**

Initial screening shows no significant cardiac concerns.

**Risk Level**: ${result.probability ? `${(result.probability * 100).toFixed(1)}%` : 'Low'}

No further analysis required. Upload a new patient file to start another assessment.`,
          });
        }
      } catch (error) {
        updateMessage(processingId, {
          type: 'bot',
          content: `Error during analysis: ${error instanceof Error ? error.message : 'Unknown error'}. Please try again.`,
          isLoading: false,
        });
      }

      setIsProcessing(false);
    } else if (phase === 'phase1_complete') {
      addMessage({
        type: 'user',
        content: `Uploaded: **${fileName}** for detailed analysis`,
      });

      const compatibleModels = detectCompatibleModels(data).filter(m => m !== 'normal_heart');
      
      if (compatibleModels.length === 0) {
        addMessage({
          type: 'bot',
          content: `**No Compatible Models**

The uploaded file doesn't match any disease detection models. Required fields:

• **Heart Attack**: age, sex, chest_pain_type, trestbps, chol, fbs, restecg, thalach, exercise_angina, oldpeak, st_slope
• **Heart Failure**: age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope
• **Hypertension**: male, age, education, current_smoker, cigs_per_day, bp_meds, prevalent_stroke, prevalent_hyp, diabetes, tot_chol, sys_bp, dia_bp, bmi, heart_rate, glucose
• **Coronary**: exang, thal, restecg, slope, thalach, chol, oldpeak, trestbps, cp, ca`,
        });
        return;
      }

      const processingId = addMessage({
        type: 'bot',
        content: `Running Phase 2: Detected ${compatibleModels.length} compatible model(s)...`,
        isLoading: true,
      });

      setIsProcessing(true);

      try {
        const results: { modelId: string; result: PredictionResponse }[] = [];
        
        for (const modelId of compatibleModels) {
          try {
            const result = await predictSingle(modelId, data);
            results.push({ modelId, result });
          } catch (err) {
            console.error(`Error running ${modelId}:`, err);
          }
        }

        if (results.length === 0) {
          updateMessage(processingId, {
            type: 'bot',
            content: `Analysis failed. Please verify data format and try again.`,
            isLoading: false,
          });
          setIsProcessing(false);
          return;
        }

        const detectedModel = results[0].modelId;
        setSuggestedModel(detectedModel);

        updateMessage(processingId, {
          type: 'bot',
          content: `**Phase 2 Complete** — ${results.length} model(s) analyzed:`,
          isLoading: false,
          predictions: results,
        });

        const positiveResults = results.filter(r => r.result.prediction === 1);

        addMessage({
          type: 'bot',
          content: `**Analysis Summary**

**Phase 1**: ${phase1Result?.prediction === 1 ? 'Elevated Risk' : 'Normal'}

**Phase 2 Results**:
${results.map(r => `• **${getModelName(r.modelId)}**: ${r.result.prediction === 1 ? 'Positive' : 'Negative'} (${r.result.probability ? `${(r.result.probability * 100).toFixed(1)}%` : 'N/A'})`).join('\n')}

${positiveResults.length > 0 
  ? `**Recommendation**: ${positiveResults.length} condition(s) detected. Consult a healthcare professional for evaluation.` 
  : 'No significant concerns detected in this analysis.'}

Upload another file for additional testing, or type "restart" for new assessment.`,
        });
      } catch (error) {
        updateMessage(processingId, {
          type: 'bot',
          content: `Error during analysis: ${error instanceof Error ? error.message : 'Unknown error'}. Please try again.`,
          isLoading: false,
        });
      }

      setIsProcessing(false);
    } else {
      addMessage({
        type: 'user',
        content: `Uploaded: **${fileName}**`,
      });
      addMessage({
        type: 'bot',
        content: 'File received. Type "restart" to begin a new assessment.',
      });
    }
  };

  const suggestModelBasedOnData = (data: Record<string, unknown>): string => {
    const keys = Object.keys(data);
    
    if (keys.some(k => k.toLowerCase().includes('troponin') || k.toLowerCase().includes('ck_mb'))) {
      return 'heart_attack';
    }
    if (keys.some(k => k.toLowerCase().includes('ejection') || k.toLowerCase().includes('sodium') || k.toLowerCase().includes('creatinine'))) {
      return 'heart_failure';
    }
    if (keys.some(k => k.toLowerCase().includes('bp') || k.toLowerCase().includes('systolic') || k.toLowerCase().includes('diastolic'))) {
      return 'hypertension';
    }
    if (keys.some(k => k.toLowerCase().includes('stenosis') || k.toLowerCase().includes('plaque'))) {
      return 'coronary';
    }
    
    return 'heart_failure';
  };

  const getModelName = (modelId: string): string => {
    const model = MODELS.find(m => m.id === modelId);
    return model?.name || modelId;
  };

  const handleSend = async () => {
    if (!inputText.trim() || isProcessing) return;

    const userMessage = inputText.trim().toLowerCase();
    setInputText('');

    addMessage({
      type: 'user',
      content: inputText.trim(),
    });

    if (userMessage === 'restart' || userMessage === 'start over' || userMessage === 'reset') {
      setPhase('initial');
      setPhase1Result(null);
      setSuggestedModel(null);
      setUploadedData(null);
      addMessage({
        type: 'bot',
        content: `**Assessment Reset**

Ready for new analysis. Upload your heart health JSON file to begin Phase 1.`,
      });
      return;
    }

    if (userMessage === 'skip' && phase === 'phase1_complete') {
      setPhase('phase2_complete');
      addMessage({
        type: 'bot',
        content: `**Assessment Complete**

Phase 2 skipped. Based on Phase 1:
${phase1Result?.prediction === 1 
  ? 'Elevated risk detected. Consider consulting a healthcare professional.' 
  : 'No immediate concerns identified.'}

Type "restart" for new assessment.`,
      });
      return;
    }

    const modelMatch = SPECIFIC_MODELS.find(m => 
      userMessage.includes(m.id) || userMessage.includes(m.name.toLowerCase())
    );

    if (modelMatch && phase === 'phase1_complete') {
      setSuggestedModel(modelMatch.id);
      addMessage({
        type: 'bot',
        content: `Selected: **${modelMatch.name}** model.

Upload the appropriate JSON file in the sidebar to proceed.`,
      });
      return;
    }

    if (phase === 'initial') {
      addMessage({
        type: 'bot',
        content: `Upload a heart health JSON file using the sidebar to begin assessment.`,
      });
    } else if (phase === 'phase1_complete') {
      addMessage({
        type: 'bot',
        content: `Phase 1 complete. Options:
• Upload file for Phase 2 analysis
• Type condition name (heart_attack, heart_failure, hypertension, coronary)
• Type "skip" to end
• Type "restart" to start over`,
      });
    } else {
      addMessage({
        type: 'bot',
        content: `Assessment complete. Type "restart" for new analysis.`,
      });
    }
  };

  const getPhaseIndicator = () => {
    switch (phase) {
      case 'initial':
        return { text: 'Phase 1: Screening', color: 'bg-blue-600' };
      case 'phase1_complete':
        return { text: 'Phase 2: Analysis', color: 'bg-amber-600' };
      case 'phase2_complete':
        return { text: 'Complete', color: 'bg-emerald-600' };
    }
  };

  if (!isClient) {
    return (
      <div className="flex h-screen bg-slate-900">
        <div className="flex-1 flex items-center justify-center">
          <div className="animate-pulse flex items-center gap-3">
            <Activity className="w-6 h-6 text-cyan-400" />
            <span className="text-slate-400">Loading HeartGuard AI...</span>
          </div>
        </div>
      </div>
    );
  }

  const phaseInfo = getPhaseIndicator();

  return (
    <div className="flex h-screen bg-slate-900">
      {/* Sidebar */}
      <AnimatePresence>
        {showSidebar && (
          <motion.div
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 300, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            className="bg-slate-800/50 border-r border-slate-700/50 overflow-hidden backdrop-blur"
          >
            <div className="p-4 h-full flex flex-col">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-base font-semibold text-slate-200 flex items-center gap-2">
                  <Activity className="w-5 h-5 text-cyan-400" />
                  HeartGuard AI
                </h2>
                <button
                  onClick={() => setShowSidebar(false)}
                  className="p-1.5 hover:bg-slate-700/50 rounded-lg transition-colors"
                >
                  <X className="w-4 h-4 text-slate-400" />
                </button>
              </div>

              {/* Phase Indicator */}
              <div className={`${phaseInfo.color} text-white px-3 py-2 rounded-lg mb-4 text-sm font-medium`}>
                {phaseInfo.text}
              </div>

              {/* Instructions */}
              <div className="bg-slate-700/30 rounded-lg p-3 mb-4 border border-slate-700/50">
                <h3 className="font-medium text-slate-300 mb-1 text-sm">
                  {phase === 'initial' && 'Upload Baseline Data'}
                  {(phase === 'phase1_complete' || phase === 'phase2_complete') && 'Upload Patient Data'}
                </h3>
                <p className="text-xs text-slate-400 leading-relaxed">
                  {phase === 'initial' && 'Upload normal heart JSON for initial screening.'}
                  {(phase === 'phase1_complete' || phase === 'phase2_complete') && 'Upload any patient data. Auto-detection enabled.'}
                </p>
              </div>

              <FileUpload 
                onFileUpload={handleFileUpload} 
                disabled={isProcessing}
              />

              {(phase === 'phase1_complete' || phase === 'phase2_complete') && (
                <button
                  onClick={() => {
                    setPhase('initial');
                    setPhase1Result(null);
                    setSuggestedModel(null);
                    setUploadedData(null);
                    setMessages([createWelcomeMessage()]);
                  }}
                  className="mt-4 flex items-center justify-center gap-2 w-full py-2.5 px-4 bg-slate-700/50 hover:bg-slate-700 text-slate-300 rounded-lg transition-colors text-sm border border-slate-600/50"
                >
                  <RotateCcw className="w-4 h-4" />
                  New Assessment
                </button>
              )}

              {/* Models List */}
              <div className="mt-auto pt-4 border-t border-slate-700/50">
                <h3 className="text-xs font-medium text-slate-500 mb-2 uppercase tracking-wide">Available Models</h3>
                <div className="space-y-1">
                  {MODELS.map(model => (
                    <div 
                      key={model.id} 
                      className={`text-xs px-2.5 py-1.5 rounded ${
                        model.id === PHASE1_MODEL 
                          ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30' 
                          : suggestedModel === model.id
                          ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                          : 'bg-slate-700/30 text-slate-400 border border-slate-700/50'
                      }`}
                    >
                      {model.name}
                      {model.id === PHASE1_MODEL && ' • Phase 1'}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-slate-800/50 border-b border-slate-700/50 px-4 py-3 flex items-center gap-4 backdrop-blur">
          {!showSidebar && (
            <button
              onClick={() => setShowSidebar(true)}
              className="p-2 hover:bg-slate-700/50 rounded-lg transition-colors"
            >
              <Menu className="w-5 h-5 text-slate-400" />
            </button>
          )}
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-lg flex items-center justify-center shadow-lg shadow-cyan-500/20">
              <Activity className="w-4 h-4 text-white" />
            </div>
            <div>
              <h1 className="font-semibold text-slate-200 text-sm">HeartGuard AI</h1>
              <p className="text-xs text-slate-500">Cardiac Health Assessment</p>
            </div>
          </div>
          <div className={`ml-auto ${phaseInfo.color} text-white text-xs px-3 py-1.5 rounded-md font-medium`}>
            {phaseInfo.text}
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          <AnimatePresence>
            {messages.map(message => (
              <ChatMessage key={message.id} message={message} />
            ))}
          </AnimatePresence>
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="bg-slate-800/50 border-t border-slate-700/50 p-4 backdrop-blur">
          <div className="max-w-4xl mx-auto flex gap-3">
            <input
              type="text"
              value={inputText}
              onChange={e => setInputText(e.target.value)}
              onKeyPress={e => e.key === 'Enter' && handleSend()}
              placeholder={
                phase === 'initial' 
                  ? "Upload file to begin..." 
                  : phase === 'phase1_complete'
                  ? "Type command or upload file..."
                  : "Type 'restart' for new assessment..."
              }
              disabled={isProcessing}
              className="flex-1 px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 focus:border-cyan-500/50 disabled:opacity-50 disabled:cursor-not-allowed text-sm"
            />
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleSend}
              disabled={isProcessing || !inputText.trim()}
              className="px-4 py-3 bg-gradient-to-r from-cyan-500 to-blue-600 text-white rounded-lg hover:from-cyan-600 hover:to-blue-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 shadow-lg shadow-cyan-500/20"
            >
              <Send className="w-4 h-4" />
            </motion.button>
          </div>
          <p className="text-center text-xs text-slate-600 mt-2">
            {phase === 'initial' && 'Phase 1: Initial screening → Phase 2: Targeted analysis'}
            {phase === 'phase1_complete' && 'Ready for Phase 2 detailed analysis'}
            {phase === 'phase2_complete' && 'Assessment complete'}
          </p>
        </div>
      </div>
    </div>
  );
}
