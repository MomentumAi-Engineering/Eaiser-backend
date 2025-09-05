import React from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { Home, ArrowLeft, Search } from 'lucide-react';

function NotFound() {
  return (
    <motion.div 
      className="min-h-screen flex items-center justify-center bg-gradient-to-br from-[#1a1c3d] via-[#2d3561] to-[#3a4175] p-4 relative overflow-hidden"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.8 }}
    >
      <div 
        className="absolute inset-0 bg-[radial-gradient(circle_at_20%_50%,rgba(120,119,198,0.1)_0%,rgba(120,119,198,0)_50%),radial-gradient(circle_at_80%_80%,rgba(255,119,198,0.1)_0%,rgba(255,119,198,0)_50%),radial-gradient(circle_at_40%_20%,rgba(255,219,112,0.1)_0%,rgba(255,219,112,0)_50%)] z-[1]"
      />
      
      <motion.div 
        className="max-w-3xl text-center z-[2] relative"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
      >
        <motion.div 
          className="text-[12rem] md:text-[8rem] font-black text-white/10 mb-4 tracking-tight select-none"
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          4<motion.span 
            animate={{ 
              rotate: [0, 10, 0, -10, 0],
              scale: [1, 1.1, 1]
            }}
            transition={{ 
              duration: 2, 
              repeat: Infinity,
              repeatDelay: 1
            }}
          >0</motion.span>4
        </motion.div>
        
        <motion.div 
          className="relative h-[300px] my-8"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.4 }}
        >
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-[3]">
            <div className="w-[100px] h-[120px] bg-gradient-to-br from-gray-200 to-white rounded-[50px_50px_30px_30px] shadow-2xl relative">
              <div className="absolute -top-[30px] left-1/2 -translate-x-1/2 w-20 h-20 bg-gradient-to-br from-gray-200 to-white rounded-full shadow-lg overflow-hidden">
                <div className="absolute top-2.5 left-2.5 w-[60px] h-[60px] bg-gradient-to-br from-[rgba(100,200,255,0.3)] to-[rgba(150,220,255,0.1)] rounded-full border-2 border-white/50" />
              </div>
              <div className="absolute top-5 right-[-20px] w-10 h-[60px] bg-gradient-to-br from-gray-300 to-gray-100 rounded-[10px] shadow-lg" />
            </div>
            <div className="absolute w-[25px] h-[70px] bg-gradient-to-br from-gray-200 to-white rounded-xl top-[30px] -left-5 -rotate-[20deg] origin-top shadow-lg" />
            <div className="absolute w-[25px] h-[70px] bg-gradient-to-br from-gray-200 to-white rounded-xl top-[30px] -right-5 rotate-[20deg] origin-top shadow-lg" />
            <div className="absolute w-[30px] h-[60px] bg-gradient-to-br from-gray-200 to-white rounded-[15px] -bottom-[50px] left-5 -rotate-[10deg] origin-top shadow-lg" />
            <div className="absolute w-[30px] h-[60px] bg-gradient-to-br from-gray-200 to-white rounded-[15px] -bottom-[50px] right-5 rotate-[10deg] origin-top shadow-lg" />
          </div>
          
          <div className="absolute bottom-5 right-[100px] w-[150px] h-[150px] bg-gradient-to-br from-red-400 to-red-300 rounded-full shadow-[0_0_30px_rgba(255,107,107,0.5)] z-[2] md:w-[100px] md:h-[100px] md:right-[50px]" />
          
          <div className="absolute inset-0 z-[1]">
            {[...Array(20)].map((_, i) => (
              <div 
                key={i} 
                className="star absolute w-[2px] h-[2px] bg-white rounded-full animate-twinkle"
                style={{ 
                  top: `${Math.random() * 100}%`, 
                  left: `${Math.random() * 100}%`,
                  animationDelay: `${Math.random() * 2}s`
                }}
              />
            ))}
          </div>
          
          <div className="absolute inset-0 z-[1]">
            {[...Array(3)].map((_, i) => (
              <div 
                key={i} 
                className="meteor absolute w-[100px] h-[2px] bg-gradient-to-r from-transparent to-white/80 -rotate-45 animate-meteor"
                style={{ 
                  top: `${Math.random() * 100}%`, 
                  left: `${Math.random() * 100}%`,
                  animationDelay: `${i * 0.5}s`
                }}
              />
            ))}
          </div>
        </motion.div>
        
        <motion.h1 
          className="text-4xl md:text-5xl font-bold text-white mb-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.6 }}
        >
          Oops! Page Not Found
        </motion.h1>
        
        <motion.p 
          className="text-lg md:text-xl text-white/70 max-w-xl mx-auto mb-8 leading-relaxed"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.7 }}
        >
          The page you are looking for might have been removed, had its name changed, 
          or is temporarily unavailable.
        </motion.p>
        
        <motion.div 
          className="flex flex-col md:flex-row justify-center gap-4 mb-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.8 }}
        >
          <Link 
            to="/" 
            className="flex items-center gap-2 px-6 py-3 bg-gradient-to-br from-indigo-500 to-purple-600 text-white rounded-full font-semibold text-base hover:-translate-y-1 hover:shadow-lg transition-all md:w-[200px] justify-center"
          >
            <Home className="w-5 h-5" />
            Back to Home
          </Link>
          <button 
            className="flex items-center gap-2 px-6 py-3 bg-white/10 text-white rounded-full font-semibold text-base border border-white/20 hover:bg-white/20 transition-all md:w-[200px] justify-center"
            onClick={() => window.history.back()}
          >
            <ArrowLeft className="w-5 h-5" />
            Go Back
          </button>
        </motion.div>
        
        <motion.div 
          className="max-w-md mx-auto"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.9 }}
        >
          <p className="text-white/70 mb-4">Or try searching for what you need:</p>
          <div className="flex items-center bg-white/10 border border-white/20 rounded-full p-2 backdrop-blur-md">
            <Search className="text-white/70 mr-2 w-5 h-5" />
            <input 
              type="text" 
              placeholder="Search our site..." 
              className="bg-transparent border-none text-white text-base w-full focus:outline-none placeholder-white/50"
            />
          </div>
        </motion.div>
      </motion.div>
      
      <style>{`
        @keyframes twinkle {
          0%, 100% { opacity: 0.2; }
          50% { opacity: 1; }
        }
        
        .animate-twinkle {
          animation: twinkle 3s infinite;
        }
        
        @keyframes meteor {
          0% { 
            transform: translateX(-100px) translateY(-100px) rotate(-45deg);
            opacity: 0;
          }
          10% { opacity: 1; }
          20%, 100% { 
            transform: translateX(300px) translateY(300px) rotate(-45deg);
            opacity: 0;
          }
        }
        
        .animate-meteor {
          animation: meteor 3s infinite;
        }
      `}</style>
    </motion.div>
  );
}

export default NotFound;