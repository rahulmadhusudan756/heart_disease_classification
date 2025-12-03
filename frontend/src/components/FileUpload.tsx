'use client';

import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, FileJson, X, CheckCircle, ArrowRight } from 'lucide-react';

interface FileUploadProps {
  onFileUpload: (data: Record<string, any>, fileName: string) => void;
  disabled?: boolean;
}

export default function FileUpload({ onFileUpload, disabled }: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<Record<string, any> | null>(null);

  const processFile = useCallback(async (file: File) => {
    setError(null);
    
    if (!file.name.endsWith('.json')) {
      setError('Please upload a JSON file');
      return;
    }

    try {
      const text = await file.text();
      const data = JSON.parse(text);
      setFile(file);
      setPreview(data);
    } catch (e) {
      setError('Invalid JSON file');
      setFile(null);
      setPreview(null);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      processFile(droppedFile);
    }
  }, [processFile]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      processFile(selectedFile);
    }
  }, [processFile]);

  const handleSubmit = () => {
    if (preview && file) {
      onFileUpload(preview, file.name);
      setFile(null);
      setPreview(null);
    }
  };

  const clearFile = () => {
    setFile(null);
    setPreview(null);
    setError(null);
  };

  return (
    <div className="w-full">
      <AnimatePresence mode="wait">
        {!file ? (
          <motion.div
            key="dropzone"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            className={`
              relative border-2 border-dashed rounded-lg p-6 transition-all cursor-pointer
              ${isDragging 
                ? 'border-cyan-400 bg-cyan-500/10' 
                : 'border-slate-600 hover:border-cyan-500/50 hover:bg-slate-800/50'
              }
              ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
            `}
          >
            <input
              type="file"
              accept=".json"
              onChange={handleFileInput}
              disabled={disabled}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
            <div className="flex flex-col items-center text-center">
              <div className={`w-11 h-11 rounded-lg flex items-center justify-center mb-3 transition-colors ${
                isDragging ? 'bg-cyan-500/20' : 'bg-slate-700/50'
              }`}>
                <Upload className={`w-5 h-5 ${isDragging ? 'text-cyan-400' : 'text-slate-400'}`} />
              </div>
              <p className="text-sm font-medium text-slate-300">
                {isDragging ? 'Drop file here' : 'Drop JSON file here'}
              </p>
              <p className="text-xs text-slate-500 mt-1">or click to browse</p>
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="preview"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="border border-slate-700 rounded-lg p-4 bg-slate-800/50"
          >
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <FileJson className="w-4 h-4 text-cyan-400" />
                <span className="font-medium text-slate-300 truncate max-w-[140px] text-sm">{file.name}</span>
                <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
              </div>
              <button
                onClick={clearFile}
                className="p-1 hover:bg-slate-700 rounded transition-colors"
              >
                <X className="w-4 h-4 text-slate-400" />
              </button>
            </div>
            
            <div className="bg-slate-900/50 rounded-md p-3 mb-3 max-h-28 overflow-auto">
              <pre className="text-xs text-slate-400 font-mono whitespace-pre-wrap">
                {JSON.stringify(preview, null, 2).slice(0, 400)}
                {JSON.stringify(preview, null, 2).length > 400 && '...'}
              </pre>
            </div>

            <button
              onClick={handleSubmit}
              disabled={disabled}
              className="w-full py-2.5 bg-gradient-to-r from-cyan-500 to-blue-600 text-white rounded-lg font-medium hover:from-cyan-600 hover:to-blue-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 text-sm"
            >
              Run Analysis
              <ArrowRight className="w-4 h-4" />
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {error && (
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-red-400 text-sm mt-2 text-center"
        >
          {error}
        </motion.p>
      )}
    </div>
  );
}
