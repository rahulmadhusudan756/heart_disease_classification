'use client';

import { motion } from 'framer-motion';
import { Heart, Check } from 'lucide-react';
import { MODELS } from '@/types';

interface ModelSelectorProps {
  selectedModels: string[];
  onSelectionChange: (models: string[]) => void;
  compatibleModels: string[];
}

export default function ModelSelector({ selectedModels, onSelectionChange, compatibleModels }: ModelSelectorProps) {
  const toggleModel = (modelId: string) => {
    if (selectedModels.includes(modelId)) {
      onSelectionChange(selectedModels.filter(id => id !== modelId));
    } else {
      onSelectionChange([...selectedModels, modelId]);
    }
  };

  const selectAll = () => {
    onSelectionChange(compatibleModels);
  };

  const clearAll = () => {
    onSelectionChange([]);
  };

  return (
    <div className="bg-white rounded-xl border border-rose-100 p-4 shadow-sm">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold text-gray-800 flex items-center gap-2">
          <Heart className="w-4 h-4 text-rose-500" />
          Select Models
        </h3>
        <div className="flex gap-2">
          <button
            onClick={selectAll}
            className="text-xs text-rose-500 hover:text-rose-600 font-medium"
          >
            Select All
          </button>
          <span className="text-gray-300">|</span>
          <button
            onClick={clearAll}
            className="text-xs text-gray-500 hover:text-gray-600 font-medium"
          >
            Clear
          </button>
        </div>
      </div>
      
      <div className="space-y-2">
        {MODELS.map((model) => {
          const isCompatible = compatibleModels.includes(model.id);
          const isSelected = selectedModels.includes(model.id);
          
          return (
            <motion.button
              key={model.id}
              onClick={() => isCompatible && toggleModel(model.id)}
              whileTap={isCompatible ? { scale: 0.98 } : {}}
              className={`
                w-full flex items-center gap-3 p-3 rounded-lg border transition-all text-left
                ${!isCompatible 
                  ? 'opacity-40 cursor-not-allowed bg-gray-50 border-gray-200' 
                  : isSelected
                    ? 'bg-rose-50 border-rose-300 shadow-sm'
                    : 'bg-white border-gray-200 hover:border-rose-200 hover:bg-rose-50/50'
                }
              `}
              disabled={!isCompatible}
            >
              <div className={`
                w-5 h-5 rounded border-2 flex items-center justify-center transition-colors
                ${isSelected ? 'bg-rose-500 border-rose-500' : 'border-gray-300'}
              `}>
                {isSelected && <Check className="w-3 h-3 text-white" />}
              </div>
              <div className="flex-1 min-w-0">
                <p className={`font-medium text-sm truncate ${isCompatible ? 'text-gray-800' : 'text-gray-500'}`}>
                  {model.name}
                </p>
                <p className="text-xs text-gray-500 truncate">{model.description}</p>
              </div>
              {!isCompatible && (
                <span className="text-xs text-gray-400 bg-gray-100 px-2 py-1 rounded">
                  Missing fields
                </span>
              )}
            </motion.button>
          );
        })}
      </div>
      
      {selectedModels.length > 0 && (
        <p className="text-xs text-gray-500 mt-3 text-center">
          {selectedModels.length} model{selectedModels.length > 1 ? 's' : ''} selected
        </p>
      )}
    </div>
  );
}
