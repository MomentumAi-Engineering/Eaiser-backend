import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence, useAnimation } from 'framer-motion';
import { useInView } from 'react-intersection-observer';
import { 
  FiUploadCloud, 
  FiAlertCircle, 
  FiCheckCircle, 
  FiLoader, 
  FiRefreshCw,
  FiMapPin,
  FiCalendar,
  FiInfo,
  FiChevronDown,
  FiChevronUp,
  FiEye,
  FiEdit,
  FiUser,
  FiClock,
  FiTag,
  FiMessageSquare,
  FiCheck,
  FiX,
  FiCamera,
  FiMap,
  FiStar,
  FiFilter,
  FiSearch,
  FiZap,
  FiTrendingUp,
  FiActivity,
  FiShield
} from 'react-icons/fi';
import UploadForm from '../components/UploadForm';

// Floating particles animation component
const FloatingParticles = () => {
  const particles = Array.from({ length: 6 }, (_, i) => i);
  
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {particles.map((particle) => (
        <motion.div
          key={particle}
          className="absolute w-2 h-2 bg-white/20 rounded-full"
          initial={{
            x: Math.random() * 100 + '%',
            y: Math.random() * 100 + '%',
          }}
          animate={{
            x: [null, Math.random() * 100 + '%'],
            y: [null, Math.random() * 100 + '%'],
          }}
          transition={{
            duration: Math.random() * 10 + 10,
            repeat: Infinity,
            repeatType: 'reverse',
            ease: 'linear',
          }}
        />
      ))}
    </div>
  );
};

const StatusMessage = ({ status }) => {
  const [isVisible, setIsVisible] = useState(true);
  const controls = useAnimation();
  
  useEffect(() => {
    if (status) {
      setIsVisible(true);
      controls.start('animate');
      
      if (!status.includes('Loading')) {
        const timer = setTimeout(() => {
          controls.start('exit').then(() => setIsVisible(false));
        }, 5000);
        return () => clearTimeout(timer);
      }
    } else {
      controls.start('exit').then(() => setIsVisible(false));
    }
  }, [status, controls]);
  
  const variants = {
    initial: { opacity: 0, y: -30, scale: 0.8, rotateX: -90 },
    animate: { 
      opacity: 1, 
      y: 0, 
      scale: 1,
      rotateX: 0,
      transition: { 
        type: 'spring', 
        stiffness: 400, 
        damping: 25,
        duration: 0.6
      }
    },
    exit: { 
      opacity: 0, 
      y: -20, 
      scale: 0.9,
      rotateX: 90,
      transition: { duration: 0.3 }
    }
  };
  
  const getStatusIcon = () => {
    if (status.includes('Error')) return <FiAlertCircle className="mr-3 text-lg" />;
    if (status.includes('Loading')) return <FiLoader className="mr-3 text-lg animate-spin" />;
    return <FiCheckCircle className="mr-3 text-lg" />;
  };
  
  const getStatusEmoji = () => {
    if (status.includes('Error')) return '🚨';
    if (status.includes('Loading')) return '⏳';
    if (status.includes('Success')) return '🎉';
    return 'ℹ️';
  };
  
  const getStatusStyles = () => {
    if (status.includes('Error')) {
      return 'glass-card border-error/30 bg-gradient-to-r from-error/10 to-error/5 text-error shadow-error/20';
    }
    if (status.includes('Loading')) {
      return 'glass-card border-primary/30 bg-gradient-to-r from-primary/10 to-primary/5 text-primary shadow-primary/20';
    }
    return 'glass-card border-success/30 bg-gradient-to-r from-success/10 to-success/5 text-success shadow-success/20';
  };
  
  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          variants={variants}
          initial="initial"
          animate={controls}
          exit="exit"
          className={`flex items-center p-5 rounded-2xl mt-8 cursor-pointer transition-all duration-500 backdrop-blur-sm border ${getStatusStyles()}`}
          role="alert"
          onClick={() => controls.start('exit').then(() => setIsVisible(false))}
          whileHover={{ 
            scale: 1.02, 
            y: -2,
            boxShadow: '0 20px 40px rgba(0,0,0,0.1)'
          }}
          whileTap={{ scale: 0.98 }}
        >
          <motion.span 
            className="text-2xl mr-4"
            animate={{ 
              rotate: [0, 10, -10, 0],
              scale: [1, 1.1, 1]
            }}
            transition={{ 
              duration: 2, 
              repeat: Infinity,
              repeatType: 'reverse'
            }}
          >
            {getStatusEmoji()}
          </motion.span>
          <div className="flex items-center font-semibold text-lg">
            <motion.div
              animate={status.includes('Loading') ? { rotate: 360 } : {}}
              transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
            >
              {getStatusIcon()}
            </motion.div>
            <motion.span
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              {status}
            </motion.span>
          </div>
          <motion.div
            className="ml-auto text-sm opacity-60 hover:opacity-100 transition-opacity"
            whileHover={{ scale: 1.1 }}
          >
            <FiX className="w-4 h-4" />
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

const IssueCard = ({ issue, index, expanded, toggleExpand }) => {
  const cardVariants = {
    hidden: { opacity: 0, y: 30, rotateY: -15 },
    visible: {
      opacity: 1,
      y: 0,
      rotateY: 0,
      transition: {
        delay: index * 0.15,
        type: 'spring',
        stiffness: 120,
        damping: 15,
        duration: 0.8
      }
    }
  };
  
  const uniqueKey = issue.id || `issue-${index}`;
  
  const getStatusColor = () => {
    switch (issue.status?.toLowerCase()) {
      case 'resolved':
        return 'bg-success/10 text-success border-success/20 shadow-success/10';
      case 'in-progress':
        return 'bg-primary/10 text-primary border-primary/20 shadow-primary/10';
      case 'rejected':
        return 'bg-error/10 text-error border-error/20 shadow-error/10';
      default:
        return 'bg-warning/10 text-warning border-warning/20 shadow-warning/10';
    }
  };
  
  const getSeverityColor = () => {
    switch (issue.severity?.toLowerCase()) {
      case 'critical':
        return 'bg-error/15 text-error border-error/30 shadow-error/20';
      case 'high':
        return 'bg-warning/15 text-warning border-warning/30 shadow-warning/20';
      case 'medium':
        return 'bg-primary/15 text-primary border-primary/30 shadow-primary/20';
      case 'low':
        return 'bg-success/15 text-success border-success/30 shadow-success/20';
      default:
        return 'bg-neutral-200/50 text-neutral-600 border-neutral-300/50';
    }
  };
  
  const getSeverityIcon = () => {
    switch (issue.severity?.toLowerCase()) {
      case 'critical':
        return <FiZap className="w-3 h-3" />;
      case 'high':
        return <FiTrendingUp className="w-3 h-3" />;
      case 'medium':
        return <FiActivity className="w-3 h-3" />;
      case 'low':
        return <FiShield className="w-3 h-3" />;
      default:
        return <FiInfo className="w-3 h-3" />;
    }
  };
  
  return (
    <motion.div
      key={uniqueKey}
      variants={cardVariants}
      initial="hidden"
      animate="visible"
      className={`glass-card rounded-3xl p-6 cursor-pointer transition-all duration-500 border backdrop-blur-md ${
        expanded 
          ? 'border-primary/40 shadow-primary/20 bg-primary/5' 
          : 'border-neutral-200/30 hover:border-primary/30'
      }`}
      onClick={toggleExpand}
      whileHover={{ 
        y: -8, 
        scale: 1.02,
        boxShadow: '0 25px 50px rgba(0,0,0,0.15)'
      }}
      whileTap={{ scale: 0.98 }}
      style={{
        background: expanded 
          ? 'linear-gradient(135deg, rgba(99, 102, 241, 0.05) 0%, rgba(168, 85, 247, 0.05) 100%)'
          : 'rgba(255, 255, 255, 0.8)'
      }}
    >
      <div className="flex items-center gap-5">
        <motion.div 
          className="w-12 h-12 rounded-2xl bg-gradient-to-br from-primary to-secondary text-white flex items-center justify-center font-bold text-lg flex-shrink-0 shadow-lg"
          whileHover={{ 
            rotate: 360,
            scale: 1.1
          }}
          transition={{ duration: 0.6 }}
        >
          <span>{index + 1}</span>
        </motion.div>
        
        <div className="flex-1 min-w-0">
          <motion.h5 
            className="text-xl font-bold text-neutral-800 mb-3 truncate"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
          >
            {issue.title || 'Untitled Issue'}
          </motion.h5>
          
          <div className="flex gap-3 flex-wrap">
            <motion.span 
              className={`px-4 py-2 rounded-xl text-sm font-semibold border backdrop-blur-sm ${getStatusColor()}`}
              whileHover={{ scale: 1.05 }}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.3 }}
            >
              {issue.status || 'Pending'}
            </motion.span>
            
            <motion.span 
              className={`px-4 py-2 rounded-xl text-sm font-semibold border backdrop-blur-sm flex items-center gap-2 ${getSeverityColor()}`}
              whileHover={{ scale: 1.05 }}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.4 }}
            >
              {getSeverityIcon()}
              {issue.severity || 'Unknown'}
            </motion.span>
          </div>
        </div>
        
        <motion.div 
          className="text-neutral-400 flex-shrink-0 p-2 rounded-full hover:bg-neutral-100/50 transition-colors"
          whileHover={{ 
            scale: 1.2,
            rotate: expanded ? 180 : 0
          }}
          animate={{ 
            rotate: expanded ? 180 : 0
          }}
          transition={{ duration: 0.3 }}
        >
          <FiChevronDown className="w-5 h-5" />
        </motion.div>
      </div>
      
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0, y: -20 }}
            animate={{ height: 'auto', opacity: 1, y: 0 }}
            exit={{ height: 0, opacity: 0, y: -20 }}
            transition={{ duration: 0.5, ease: 'easeInOut' }}
            className="mt-6 pt-6 border-t border-neutral-200/50"
          >
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
              <motion.div 
                className="flex items-start gap-3 p-4 rounded-2xl bg-neutral-50/50 backdrop-blur-sm border border-neutral-200/30"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.1 }}
              >
                <div className="p-2 rounded-xl bg-primary/10">
                  <FiMapPin className="text-primary w-4 h-4" />
                </div>
                <div>
                  <p className="text-xs font-semibold text-neutral-500 uppercase tracking-wide mb-1">Location</p>
                  <p className="text-sm font-medium text-neutral-700">{issue.location || 'No location provided'}</p>
                </div>
              </motion.div>
              
              <motion.div 
                className="flex items-start gap-3 p-4 rounded-2xl bg-neutral-50/50 backdrop-blur-sm border border-neutral-200/30"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.2 }}
              >
                <div className="p-2 rounded-xl bg-secondary/10">
                  <FiCalendar className="text-secondary w-4 h-4" />
                </div>
                <div>
                  <p className="text-xs font-semibold text-neutral-500 uppercase tracking-wide mb-1">Date</p>
                  <p className="text-sm font-medium text-neutral-700">{issue.date ? new Date(issue.date).toLocaleDateString() : 'No date provided'}</p>
                </div>
              </motion.div>
              
              <motion.div 
                className="flex items-start gap-3 p-4 rounded-2xl bg-neutral-50/50 backdrop-blur-sm border border-neutral-200/30"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.3 }}
              >
                <div className="p-2 rounded-xl bg-success/10">
                  <FiUser className="text-success w-4 h-4" />
                </div>
                <div>
                  <p className="text-xs font-semibold text-neutral-500 uppercase tracking-wide mb-1">Reporter</p>
                  <p className="text-sm font-medium text-neutral-700">{issue.reporter || 'Anonymous'}</p>
                </div>
              </motion.div>
              
              <motion.div 
                className="flex items-start gap-3 p-4 rounded-2xl bg-neutral-50/50 backdrop-blur-sm border border-neutral-200/30"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.4 }}
              >
                <div className="p-2 rounded-xl bg-warning/10">
                  <FiClock className="text-warning w-4 h-4" />
                </div>
                <div>
                  <p className="text-xs font-semibold text-neutral-500 uppercase tracking-wide mb-1">Timestamp</p>
                  <p className="text-sm font-medium text-neutral-700">{issue.timestamp ? new Date(issue.timestamp).toLocaleString() : 'No timestamp'}</p>
                </div>
              </motion.div>
            </div>
            
            <motion.div 
              className="mb-6 p-5 rounded-2xl bg-gradient-to-br from-neutral-50/80 to-neutral-100/50 backdrop-blur-sm border border-neutral-200/30"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
            >
              <div className="flex items-start gap-3">
                <div className="p-2 rounded-xl bg-primary/10">
                  <FiInfo className="text-primary w-4 h-4" />
                </div>
                <div className="flex-1">
                  <p className="text-xs font-semibold text-neutral-500 uppercase tracking-wide mb-2">Description</p>
                  <p className="text-neutral-700 leading-relaxed">{issue.description || 'No description provided'}</p>
                </div>
              </div>
            </motion.div>
            
            {issue.category && (
              <motion.div 
                className="mb-6 p-4 rounded-2xl bg-neutral-50/50 backdrop-blur-sm border border-neutral-200/30 inline-block"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.6 }}
              >
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-xl bg-secondary/10">
                    <FiTag className="text-secondary w-4 h-4" />
                  </div>
                  <div>
                    <p className="text-xs font-semibold text-neutral-500 uppercase tracking-wide mb-1">Category</p>
                    <p className="text-sm font-medium text-neutral-700">{issue.category}</p>
                  </div>
                </div>
              </motion.div>
            )}
            
            {issue.image && (
              <motion.div 
                className="mb-6 rounded-3xl overflow-hidden shadow-lg border border-neutral-200/30"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.7 }}
                whileHover={{ scale: 1.02 }}
              >
                <img 
                  src={issue.image} 
                  alt="Issue" 
                  className="w-full h-auto"
                />
              </motion.div>
            )}
            
            <motion.div 
              className="flex flex-col sm:flex-row gap-4"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.8 }}
            >
              <motion.button 
                className="btn-secondary flex items-center justify-center gap-3 px-6 py-3 rounded-2xl font-semibold transition-all duration-300"
                whileHover={{ scale: 1.05, y: -2 }}
                whileTap={{ scale: 0.95 }}
              >
                <FiEye className="w-4 h-4" />
                <span>View Details</span>
              </motion.button>
              
              <motion.button 
                className="btn-primary flex items-center justify-center gap-3 px-6 py-3 rounded-2xl font-semibold transition-all duration-300"
                whileHover={{ scale: 1.05, y: -2 }}
                whileTap={{ scale: 0.95 }}
              >
                <FiEdit className="w-4 h-4" />
                <span>Edit Issue</span>
              </motion.button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

function ReportIssue() {
  const [status, setStatus] = useState('');
  const [issues, setIssues] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [expandedIssue, setExpandedIssue] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');
  const [showFilters, setShowFilters] = useState(false);
  
  const controls = useAnimation();
  const [ref, isInView] = useInView({ once: false, amount: 0.1 });
  
  useEffect(() => {
    if (isInView) {
      controls.start('visible');
    }
  }, [isInView, controls]);
  
  const fetchIssues = async () => {
    setIsLoading(true);
    setStatus('🔍 Loading issues... Please wait');
    try {
      await new Promise(resolve => setTimeout(resolve, 800));
      const response = await fetch('http://localhost:10000/api/issues');
      const data = await response.json();
      const issuesWithIds = data.map((issue, index) => ({
        ...issue,
        id: issue.id || `issue-${index}`,
        title: issue.title || `Issue #${index + 1}`,
        location: issue.location || 'Unknown location',
        date: issue.date || new Date().toISOString(),
        description: issue.description || 'No description provided',
        reporter: issue.reporter || 'Anonymous',
        timestamp: issue.timestamp || new Date().toISOString(),
        status: issue.status || 'pending',
        severity: issue.severity || 'medium',
        category: issue.category || 'general'
      }));
      setIssues(issuesWithIds);
      setStatus('✅ Successfully loaded issues!');
    } catch (error) {
      setStatus('🚨 Error fetching issues. Please try again later.');
      console.error('Error:', error);
    } finally {
      setIsLoading(false);
    }
  };
  
  const toggleExpandIssue = (id) => {
    setExpandedIssue(expandedIssue === id ? null : id);
  };
  
  const filteredIssues = issues.filter(issue => {
    const matchesSearch = searchTerm === '' || 
      issue.title?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      issue.description?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      issue.location?.toLowerCase().includes(searchTerm.toLowerCase());
    
    const matchesStatus = filterStatus === 'all' || issue.status === filterStatus;
    
    return matchesSearch && matchesStatus;
  });
  
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        when: "beforeChildren"
      }
    }
  };
  
  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        type: "spring",
        stiffness: 100,
        damping: 10
      }
    }
  };
  
  const statusOptions = [
    { value: 'all', label: 'All Statuses' },
    { value: 'pending', label: 'Pending' },
    { value: 'in-progress', label: 'In Progress' },
    { value: 'resolved', label: 'Resolved' },
    { value: 'rejected', label: 'Rejected' }
  ];
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-neutral-50 via-neutral-100 to-neutral-200 relative overflow-hidden">
      <FloatingParticles />
      
      {/* Background Pattern */}
      <div className="absolute inset-0 opacity-30">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(var(--primary-rgb),0.1),transparent_50%)]" />
        <div className="absolute inset-0 bg-[linear-gradient(45deg,transparent_49%,rgba(var(--secondary-rgb),0.05)_50%,transparent_51%)] bg-[length:20px_20px]" />
      </div>
      
      <div className="relative z-10 py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial="hidden"
            animate={controls}
            variants={containerVariants}
            className="text-center mb-16"
          >
            <motion.div
              className="inline-block mb-6"
              initial={{ scale: 0, rotate: -180 }}
              animate={{ scale: 1, rotate: 0 }}
              transition={{ duration: 0.8, ease: 'easeOut' }}
            >
              <div className="w-20 h-20 mx-auto rounded-3xl bg-gradient-to-br from-primary to-secondary p-4 shadow-2xl">
                <FiUploadCloud className="w-full h-full text-white" />
              </div>
            </motion.div>
            
            <motion.h1 
              className="text-5xl md:text-7xl font-bold bg-gradient-to-r from-neutral-900 via-neutral-700 to-neutral-900 bg-clip-text text-transparent mb-6"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2, duration: 0.8 }}
            >
              Report a Community Issue
            </motion.h1>
            
            <motion.p 
              className="text-xl md:text-2xl text-neutral-600 max-w-4xl mx-auto leading-relaxed"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4, duration: 0.8 }}
            >
              Snap a photo 📸 of the problem and our AI will analyze it automatically!
            </motion.p>
          </motion.div>

          <motion.div
            variants={itemVariants}
            className="glass-card p-8 rounded-3xl shadow-2xl border border-neutral-200/50 mb-16"
            whileHover={{ y: -5, scale: 1.02 }}
            transition={{ duration: 0.3 }}
          >
            <UploadForm setStatus={setStatus} fetchIssues={fetchIssues} />
            
            <StatusMessage status={status} />
          </motion.div>
          
          {issues.length > 0 && (
            <motion.div
              className="glass-card p-8 rounded-3xl shadow-2xl border border-neutral-200/50 mb-16"
              variants={itemVariants}
              initial="hidden"
              animate={controls}
            >
              <motion.div 
                className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-4"
                variants={itemVariants}
              >
                <div className="flex items-center gap-4">
                  <div className="p-3 rounded-2xl bg-primary/10">
                    <FiRefreshCw className="w-6 h-6 text-primary" />
                  </div>
                  <h3 className="text-3xl font-bold text-neutral-900">
                    📋 Recent Community Issues ({issues.length})
                  </h3>
                </div>
                
                <div className="flex flex-col md:flex-row gap-4 items-start md:items-center w-full md:w-auto">
                  <div className="relative w-full md:w-64">
                    <FiSearch className="absolute left-3 top-1/2 -translate-y-1/2 text-neutral-400" />
                    <input
                      type="text"
                      placeholder="Search issues..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="w-full pl-10 pr-4 py-3 border border-neutral-300/50 rounded-2xl text-sm focus:outline-none focus:border-primary/50 focus:ring-2 focus:ring-primary/20 transition backdrop-blur-sm bg-white/80"
                    />
                  </div>
                  
                  <motion.button 
                    className="flex items-center gap-2 px-4 py-3 bg-neutral-50/80 border border-neutral-300/50 rounded-2xl text-sm text-neutral-600 hover:bg-neutral-100/80 transition backdrop-blur-sm"
                    onClick={() => setShowFilters(!showFilters)}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <FiFilter />
                    <span>Filters</span>
                  </motion.button>
                </div>
              </motion.div>
              
              <AnimatePresence>
                {showFilters && (
                  <motion.div 
                    className="mb-8 p-6 bg-neutral-50/80 rounded-2xl border border-neutral-300/50 backdrop-blur-sm"
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    <div className="flex flex-col md:flex-row items-start md:items-center gap-4">
                      <span className="font-semibold text-neutral-700 whitespace-nowrap">Filter by status:</span>
                      <div className="flex flex-wrap gap-3">
                        {statusOptions.map(option => (
                          <motion.button
                            key={option.value}
                            className={`px-4 py-2 rounded-xl text-sm font-medium border transition backdrop-blur-sm ${
                              filterStatus === option.value 
                                ? 'bg-primary text-white border-primary shadow-primary/20' 
                                : 'bg-white/80 text-neutral-600 border-neutral-300/50 hover:border-primary/30'
                            }`}
                            onClick={() => setFilterStatus(option.value)}
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                          >
                            {option.label}
                          </motion.button>
                        ))}
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
              
              {filteredIssues.length === 0 ? (
                <motion.div 
                  className="flex flex-col items-center justify-center p-16 text-center text-neutral-400"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                >
                  <motion.div 
                    className="text-6xl mb-6 opacity-50"
                    animate={{ 
                      rotate: [0, 10, -10, 0],
                      scale: [1, 1.1, 1]
                    }}
                    transition={{ 
                      duration: 3, 
                      repeat: Infinity,
                      repeatType: 'reverse'
                    }}
                  >
                    <FiSearch />
                  </motion.div>
                  <h4 className="text-xl text-neutral-600 mb-3 font-semibold">No issues found</h4>
                  <p className="text-neutral-500">Try adjusting your search or filter criteria</p>
                </motion.div>
              ) : (
                <motion.div className="flex flex-col gap-6" variants={containerVariants}>
                  {filteredIssues.slice(0, 5).map((issue, index) => (
                    <IssueCard
                      key={issue.id}
                      issue={issue}
                      index={index}
                      expanded={expandedIssue === issue.id}
                      toggleExpand={() => toggleExpandIssue(issue.id)}
                    />
                  ))}
                </motion.div>
              )}
              
              <div className="mt-8 pt-8 border-t border-neutral-200/50 text-center">
                <motion.button
                  onClick={fetchIssues}
                  disabled={isLoading}
                  whileTap={{ scale: 0.95 }}
                  whileHover={{ scale: 1.05, y: -2 }}
                  className="inline-flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-primary to-secondary text-white rounded-2xl font-semibold shadow-2xl hover:shadow-primary/25 transition-all duration-300 disabled:opacity-70 disabled:cursor-not-allowed backdrop-blur-sm"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.4 }}
                >
                  {isLoading ? (
                    <>
                      <FiLoader className="animate-spin w-5 h-5" />
                      <span>Refreshing... ♻️</span>
                    </>
                  ) : (
                    <>
                      <FiRefreshCw className="w-5 h-5" />
                      <span>Refresh Issues 🔄</span>
                    </>
                  )}
                </motion.button>
              </div>
            </motion.div>
          )}
        </div>
      </div>
      
      <style>{`
        /* Custom styles for elements Tailwind can't fully cover */
        .glass-card {
          background: rgba(255, 255, 255, 0.9);
          backdrop-filter: blur(20px);
          -webkit-backdrop-filter: blur(20px);
        }
        
        .custom-scrollbar {
          scrollbar-width: thin;
          scrollbar-color: rgba(99, 102, 241, 0.3) transparent;
        }
        
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(99, 102, 241, 0.3);
          border-radius: 3px;
        }
        
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(99, 102, 241, 0.5);
        }
        
        .btn-primary {
          background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
          color: white;
          border: none;
        }
        
        .btn-secondary {
          background: rgba(255, 255, 255, 0.9);
          color: #374151;
          border: 1px solid rgba(209, 213, 219, 0.5);
        }
      `}</style>
    </div>
  );
}

export default ReportIssue;