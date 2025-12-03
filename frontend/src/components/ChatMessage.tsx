'use client';

import { motion } from 'framer-motion';
import { Activity, Bot, User, AlertCircle, CheckCircle, AlertTriangle } from 'lucide-react';
import { Message, ModelPrediction, MODELS } from '@/types';

interface ChatMessageProps {
  message: Message;
}

// Parse **bold** markdown syntax to JSX
function parseContent(content: string): React.ReactNode[] {
  const parts = content.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, index) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={index} className="font-semibold text-slate-100">{part.slice(2, -2)}</strong>;
    }
    return part;
  });
}

function RiskBadge({ riskLevel }: { riskLevel: string }) {
  const getColor = () => {
    switch (riskLevel.toLowerCase()) {
      case 'low risk':
        return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
      case 'mild risk':
        return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
      case 'moderate risk':
        return 'bg-orange-500/20 text-orange-400 border-orange-500/30';
      case 'high risk':
        return 'bg-red-500/20 text-red-400 border-red-500/30';
      default:
        return 'bg-slate-500/20 text-slate-400 border-slate-500/30';
    }
  };

  const getIcon = () => {
    switch (riskLevel.toLowerCase()) {
      case 'low risk':
        return <CheckCircle className="w-3.5 h-3.5" />;
      case 'high risk':
        return <AlertCircle className="w-3.5 h-3.5" />;
      default:
        return <AlertTriangle className="w-3.5 h-3.5" />;
    }
  };

  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium border ${getColor()}`}>
      {getIcon()}
      {riskLevel}
    </span>
  );
}

function PredictionCard({ prediction }: { prediction: ModelPrediction }) {
  const modelName = MODELS.find(m => m.id === prediction.modelId)?.name || prediction.result.model_name;
  const result = prediction.result;
  
  return (
    <div className="bg-slate-800/50 backdrop-blur rounded-lg p-4 border border-slate-700/50">
      <div className="flex items-center justify-between mb-3">
        <h4 className="font-medium text-slate-200 flex items-center gap-2">
          <Activity className="w-4 h-4 text-cyan-400" />
          {modelName}
        </h4>
        <RiskBadge riskLevel={result.risk_level} />
      </div>
      <div className="grid grid-cols-2 gap-2 text-sm">
        <div className="bg-slate-900/50 rounded-md p-2.5">
          <span className="text-slate-500 text-xs">Prediction</span>
          <div className={`font-semibold mt-0.5 ${result.prediction === 1 ? 'text-red-400' : 'text-emerald-400'}`}>
            {result.prediction === 1 ? 'Positive' : 'Negative'}
          </div>
        </div>
        {result.probability !== null && (
          <div className="bg-slate-900/50 rounded-md p-2.5">
            <span className="text-slate-500 text-xs">Confidence</span>
            <div className="font-semibold mt-0.5 text-slate-200">
              {(result.probability * 100).toFixed(1)}%
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default function ChatMessage({ message }: ChatMessageProps) {
  const isBot = message.type === 'bot';
  const isSystem = message.type === 'system';

  if (message.isLoading) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex gap-3 mb-4"
      >
        <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
          <Bot className="w-4 h-4 text-white" />
        </div>
        <div className="bg-slate-800/80 backdrop-blur rounded-lg rounded-tl-none p-4 border border-slate-700/50 max-w-[80%]">
          <div className="flex items-center gap-3">
            <div className="flex gap-1">
              <span className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
              <span className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
              <span className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
            </div>
            <span className="text-slate-400 text-sm">Processing analysis...</span>
          </div>
        </div>
      </motion.div>
    );
  }

  if (isSystem) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="flex justify-center mb-4"
      >
        <div className="bg-slate-800/60 text-slate-400 rounded-full px-4 py-2 text-sm border border-slate-700/50">
          {message.content}
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`flex gap-3 mb-4 ${isBot ? '' : 'flex-row-reverse'}`}
    >
      <div className={`w-9 h-9 rounded-lg flex items-center justify-center shadow-lg shrink-0 ${
        isBot 
          ? 'bg-gradient-to-br from-cyan-500 to-blue-600 shadow-cyan-500/20' 
          : 'bg-gradient-to-br from-violet-500 to-purple-600 shadow-violet-500/20'
      }`}>
        {isBot ? <Bot className="w-4 h-4 text-white" /> : <User className="w-4 h-4 text-white" />}
      </div>
      <div className={`max-w-[80%] ${isBot ? '' : 'text-right'}`}>
        <div className={`rounded-lg p-4 ${
          isBot 
            ? 'bg-slate-800/80 backdrop-blur rounded-tl-none border border-slate-700/50' 
            : 'bg-gradient-to-br from-violet-600 to-purple-700 rounded-tr-none'
        }`}>
          <p className={`text-sm whitespace-pre-wrap leading-relaxed ${isBot ? 'text-slate-300' : 'text-white'}`}>
            {parseContent(message.content)}
          </p>
        </div>
        
        {message.predictions && message.predictions.length > 0 && (
          <div className="mt-3 space-y-2">
            {message.predictions.map((pred, idx) => (
              <PredictionCard key={idx} prediction={pred} />
            ))}
          </div>
        )}
        
        <span className={`text-xs text-slate-500 mt-1.5 block ${isBot ? '' : 'text-right'}`}>
          {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </span>
      </div>
    </motion.div>
  );
}
