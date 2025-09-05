import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence, useAnimation } from 'framer-motion';
import { useInView } from 'react-intersection-observer';
import { 
  FiAlertCircle, 
  FiCheckCircle, 
  FiRefreshCw,
  FiFilter,
  FiSearch,
  FiChevronDown,
  FiChevronUp,
  FiMapPin,
  FiCalendar,
  FiClock,
  FiTrendingUp,
  FiUser,
  FiTag,
  FiMessageSquare,
  FiEye,
  FiEdit,
  FiWifi,
  FiWifiOff,
  FiSettings,
  FiDownload,
  FiGrid,
  FiList,
  FiBarChart2,
  FiActivity,
  FiCheck,
  FiX,
  FiPlus,
  FiBell
} from 'react-icons/fi';
import StatusDisplay from '../components/StatusDisplay';
import ReportDisplay from '../components/ReportDisplay';

const IssueCard = ({ issue, index, expanded, toggleExpand }) => {
  const getStatusColor = (status) => {
    switch (status?.toLowerCase()) {
      case 'resolved': return 'bg-green-100 text-green-800 border-green-200';
      case 'in-progress': return 'bg-blue-100 text-blue-800 border-blue-200';
      case 'rejected': return 'bg-red-100 text-red-800 border-red-200';
      default: return 'bg-yellow-100 text-yellow-800 border-yellow-200';
    }
  };

  const getPriorityColor = (priority) => {
    switch (priority?.toLowerCase()) {
      case 'high': return 'bg-red-100 text-red-800 border-red-200';
      case 'medium': return 'bg-orange-100 text-orange-800 border-orange-200';
      case 'low': return 'bg-green-100 text-green-800 border-green-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getStatusIcon = (status) => {
    switch (status?.toLowerCase()) {
      case 'resolved': return <FiCheckCircle className="text-green-500" />;
      case 'in-progress': return <FiClock className="text-blue-500" />;
      case 'rejected': return <FiX className="text-red-500" />;
      default: return <FiAlertCircle className="text-yellow-500" />;
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1, type: 'spring', stiffness: 100 }}
      className={`bg-white rounded-xl shadow-md overflow-hidden transition-all duration-300 border border-gray-200 ${expanded ? 'border-indigo-500' : ''}`}
      onClick={toggleExpand}
      whileHover={{ y: -3 }}
      whileTap={{ scale: 0.98 }}
    >
      <div className="flex justify-between items-start p-5 flex-col md:flex-row">
        <div className="flex flex-col gap-2">
          <span className={`inline-block px-3 py-1 rounded-full text-xs font-medium ${getPriorityColor(issue.priority)}`}>
            {issue.priority}
          </span>
          <h5 className="text-lg font-semibold text-gray-800 m-0">{issue.title}</h5>
        </div>
        <div className="text-gray-400 mt-2 md:mt-0">
          {expanded ? <FiChevronUp /> : <FiChevronDown />}
        </div>
      </div>
      
      <div className="flex flex-col md:flex-row gap-4 px-5 pb-5">
        <div className="flex items-center gap-2 text-sm text-gray-600">
          <FiMapPin className="text-indigo-500" />
          <span>{issue.location}</span>
        </div>
        <div className="flex items-center gap-2 text-sm text-gray-600">
          <FiCalendar className="text-indigo-500" />
          <span>{new Date(issue.date).toLocaleDateString()}</span>
        </div>
      </div>
      
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="px-5 pb-5 border-t border-gray-200"
          >
            <div className="mb-4">
              <div className="flex justify-between items-center mb-2">
                <h6 className="text-sm font-semibold text-gray-600 m-0">Description</h6>
                <span className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium ${getStatusColor(issue.status)}`}>
                  {getStatusIcon(issue.status)}
                  {issue.status}
                </span>
              </div>
              <p className="text-gray-600 leading-relaxed">{issue.description}</p>
            </div>
            
            {issue.image && (
              <div className="mb-4">
                <h6 className="text-sm font-semibold text-gray-600 mb-2">Image</h6>
                <img 
                  src={issue.image} 
                  alt="Issue" 
                  className="w-full h-auto rounded-lg"
                />
              </div>
            )}
            
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center mt-4 gap-4">
              <span className="flex items-center gap-2 text-xs text-gray-500">
                <FiClock />
                Reported {formatTimeAgo(issue.date)}
              </span>
              <div className="flex gap-2">
                <button className="flex items-center gap-2 px-3 py-1 bg-gray-50 border border-gray-200 rounded-md text-xs font-medium text-gray-600 hover:bg-gray-100 transition">
                  <FiEye />
                  <span>View</span>
                </button>
                <button className="flex items-center gap-2 px-3 py-1 bg-gray-50 border border-gray-200 rounded-md text-xs font-medium text-gray-600 hover:bg-gray-100 transition">
                  <FiEdit />
                  <span>Edit</span>
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

function ViewIssues() {
  const [issues, setIssues] = useState([]);
  const [filteredIssues, setFilteredIssues] = useState([]);
  const [loading, setLoading] = useState(true);
  const [realTimeEnabled, setRealTimeEnabled] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [filters, setFilters] = useState({
    status: 'all',
    priority: 'all',
    sort: 'newest'
  });
  const [viewMode, setViewMode] = useState('cards'); // 'cards' or 'table'
  const [expandedIssue, setExpandedIssue] = useState(null);
  const [notificationCount, setNotificationCount] = useState(0);
  const [lastUpdated, setLastUpdated] = useState(new Date());
  
  const controls = useAnimation();
  const [ref, isInView] = useInView({ once: false, amount: 0.1 });
  const wsRef = useRef(null);
  
  useEffect(() => {
    if (realTimeEnabled) {
      try {
        const interval = setInterval(() => {
          fetchIssues(false);
        }, 10000);
        return () => clearInterval(interval);
      } catch (error) {
        console.error('WebSocket connection failed:', error);
        setRealTimeEnabled(false);
      }
    }
  }, [realTimeEnabled]);
  
  useEffect(() => {
    if (isInView) {
      controls.start('visible');
    }
  }, [isInView, controls]);
  
  const fetchIssues = async (showLoading = true) => {
    if (showLoading) setLoading(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 800));
      const response = await fetch('http://localhost:10000/api/issues');
      const data = await response.json();
      
      if (issues.length > 0 && data.length > issues.length) {
        const newIssuesCount = data.length - issues.length;
        setNotificationCount(prev => prev + newIssuesCount);
      }
      
      setIssues(data);
      setFilteredIssues(data);
      setLastUpdated(new Date());
    } catch (error) {
      console.error('Error fetching issues:', error);
    } finally {
      if (showLoading) setLoading(false);
    }
  };
  
  useEffect(() => {
    fetchIssues();
  }, []);
  
  useEffect(() => {
    let results = [...issues];
    
    if (searchTerm) {
      results = results.filter(issue =>
        issue.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
        issue.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
        issue.location.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }
    
    if (filters.status !== 'all') {
      results = results.filter(issue => issue.status === filters.status);
    }
    
    if (filters.priority !== 'all') {
      results = results.filter(issue => issue.priority === filters.priority);
    }
    
    if (filters.sort === 'newest') {
      results.sort((a, b) => new Date(b.date) - new Date(a.date));
    } else if (filters.sort === 'oldest') {
      results.sort((a, b) => new Date(a.date) - new Date(a.date));
    } else if (filters.sort === 'priority') {
      const priorityOrder = { high: 3, medium: 2, low: 1 };
      results.sort((a, b) => priorityOrder[b.priority] - priorityOrder[a.priority]);
    }
    
    setFilteredIssues(results);
  }, [issues, searchTerm, filters]);
  
  const toggleExpandIssue = (id) => {
    setExpandedIssue(expandedIssue === id ? null : id);
  };
  
  const toggleRealTime = () => {
    setRealTimeEnabled(!realTimeEnabled);
    if (!realTimeEnabled) {
      setNotificationCount(0);
    }
  };
  
  const clearNotifications = () => {
    setNotificationCount(0);
  };
  
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
  
  const statsVariants = {
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
  
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="min-h-screen bg-gradient-to-br from-gray-100 to-gray-200 py-8 px-4"
      ref={ref}
    >
      <div className="container mx-auto max-w-7xl">
        <motion.div
          className="mb-8"
          variants={containerVariants}
          initial="hidden"
          animate={controls}
        >
          <div className="flex flex-col md:flex-row justify-between items-start gap-4 mb-8">
            <div>
              <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-2">Community Issues Dashboard</h1>
              <p className="text-gray-600">Track and manage all reported community issues in real-time</p>
            </div>
            
            <div className="flex items-center gap-4">
              <div className="flex flex-col items-end">
                <button 
                  className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium border border-gray-200 transition ${realTimeEnabled ? 'bg-teal-100 border-teal-200 text-teal-800' : 'bg-white text-gray-600'}`}
                  onClick={toggleRealTime}
                >
                  {realTimeEnabled ? <FiWifi /> : <FiWifiOff />}
                  <span>Real-time {realTimeEnabled ? 'On' : 'Off'}</span>
                </button>
                {realTimeEnabled && (
                  <div className="flex items-center gap-1 text-xs text-gray-500 mt-1">
                    <FiClock />
                    <span>Updated: {formatTimeAgo(lastUpdated)}</span>
                  </div>
                )}
              </div>
              
              <div className="relative">
                <button className="w-10 h-10 rounded-full bg-white border border-gray-200 flex items-center justify-center hover:bg-gray-50 transition" onClick={clearNotifications}>
                  <FiBell />
                  {notificationCount > 0 && (
                    <span className="absolute -top-1 -right-1 w-5 h-5 rounded-full bg-red-500 text-white text-xs font-semibold flex items-center justify-center">{notificationCount}</span>
                  )}
                </button>
              </div>
              
              <button className="w-10 h-10 rounded-full bg-white border border-gray-200 flex items-center justify-center hover:bg-gray-50 transition" onClick={() => fetchIssues()}>
                <FiRefreshCw className={loading ? 'animate-spin' : ''} />
              </button>
              
              <button className="w-10 h-10 rounded-full bg-white border border-gray-200 flex items-center justify-center hover:bg-gray-50 transition">
                <FiSettings />
              </button>
            </div>
          </div>
          
          <motion.div 
            className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-8"
            variants={containerVariants}
          >
            <motion.div className="bg-white rounded-xl shadow-md p-6 flex items-center gap-4 hover:-translate-y-1 transition" variants={statsVariants}>
              <div className="w-12 h-12 rounded-xl bg-red-100 text-red-500 flex items-center justify-center text-xl">
                <FiAlertCircle />
              </div>
              <div>
                <h3 className="text-xl font-bold text-gray-800">{issues.filter(i => i.status === 'open').length}</h3>
                <p className="text-gray-500">Open Issues</p>
              </div>
            </motion.div>
            
            <motion.div className="bg-white rounded-xl shadow-md p-6 flex items-center gap-4 hover:-translate-y-1 transition" variants={statsVariants}>
              <div className="w-12 h-12 rounded-xl bg-green-100 text-green-500 flex items-center justify-center text-xl">
                <FiCheckCircle />
              </div>
              <div>
                <h3 className="text-xl font-bold text-gray-800">{issues.filter(i => i.status === 'resolved').length}</h3>
                <p className="text-gray-500">Resolved</p>
              </div>
            </motion.div>
            
            <motion.div className="bg-white rounded-xl shadow-md p-6 flex items-center gap-4 hover:-translate-y-1 transition" variants={statsVariants}>
              <div className="w-12 h-12 rounded-xl bg-orange-100 text-orange-500 flex items-center justify-center text-xl">
                <FiTrendingUp />
              </div>
              <div>
                <h3 className="text-xl font-bold text-gray-800">{issues.filter(i => i.priority === 'high').length}</h3>
                <p className="text-gray-500">High Priority</p>
              </div>
            </motion.div>
            
            <motion.div className="bg-white rounded-xl shadow-md p-6 flex items-center gap-4 hover:-translate-y-1 transition" variants={statsVariants}>
              <div className="w-12 h-12 rounded-xl bg-blue-100 text-blue-500 flex items-center justify-center text-xl">
                <FiActivity />
              </div>
              <div>
                <h3 className="text-xl font-bold text-gray-800">
                  {issues.length > 0 ? 
                    formatTimeAgo(issues.reduce((a, b) => 
                      new Date(a.date) > new Date(b.date) ? a : b
                    ).date) : 'N/A'}
                </h3>
                <p className="text-gray-500">Last Reported</p>
              </div>
            </motion.div>
          </motion.div>
          
          <motion.div 
            className="bg-white rounded-xl shadow-md p-6 mb-8"
            variants={statsVariants}
          >
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold text-gray-800 m-0">Filters & Search</h3>
              <div className="flex bg-gray-50 rounded-lg p-1">
                <button 
                  className={`w-9 h-9 rounded-md flex items-center justify-center ${viewMode === 'cards' ? 'bg-white text-gray-800 shadow-sm' : 'text-gray-500'}`}
                  onClick={() => setViewMode('cards')}
                >
                  <FiGrid />
                </button>
                <button 
                  className={`w-9 h-9 rounded-md flex items-center justify-center ${viewMode === 'table' ? 'bg-white text-gray-800 shadow-sm' : 'text-gray-500'}`}
                  onClick={() => setViewMode('table')}
                >
                  <FiList />
                </button>
              </div>
            </div>
            
            <div className="flex flex-col md:flex-row gap-4 flex-wrap">
              <div className="relative flex-1 min-w-[250px]">
                <FiSearch className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search issues by title, description, or location..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full pl-10 pr-4 py-3 border border-gray-200 rounded-lg text-sm focus:outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100 transition"
                />
              </div>
              
              <div className="flex flex-wrap gap-4">
                <select
                  className="px-4 py-3 border border-gray-200 rounded-lg text-sm bg-white focus:outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100 transition"
                  value={filters.status}
                  onChange={(e) => setFilters({...filters, status: e.target.value})}
                >
                  <option value="all">All Statuses</option>
                  <option value="open">Open</option>
                  <option value="in-progress">In Progress</option>
                  <option value="resolved">Resolved</option>
                </select>
                
                <select
                  className="px-4 py-3 border border-gray-200 rounded-lg text-sm bg-white focus:outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100 transition"
                  value={filters.priority}
                  onChange={(e) => setFilters({...filters, priority: e.target.value})}
                >
                  <option value="all">All Priorities</option>
                  <option value="high">High</option>
                  <option value="medium">Medium</option>
                  <option value="low">Low</option>
                </select>
                
                <select
                  className="px-4 py-3 border border-gray-200 rounded-lg text-sm bg-white focus:outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100 transition"
                  value={filters.sort}
                  onChange={(e) => setFilters({...filters, sort: e.target.value})}
                >
                  <option value="newest">Newest First</option>
                  <option value="oldest">Oldest First</option>
                  <option value="priority">By Priority</option>
                </select>
                
                <button className="flex items-center gap-2 px-4 py-3 bg-indigo-500 text-white rounded-lg text-sm font-medium hover:bg-indigo-600 transition">
                  <FiDownload />
                  <span>Export</span>
                </button>
              </div>
            </div>
          </motion.div>
          
          <AnimatePresence>
            {loading && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex flex-col items-center justify-center p-12 bg-white rounded-xl shadow-md text-center"
              >
                <div className="w-12 h-12 border-4 border-gray-200 border-t-indigo-500 rounded-full animate-spin mb-4"></div>
                <p className="text-gray-600">Loading community issues...</p>
              </motion.div>
            )}
          </AnimatePresence>
          
          {!loading && (
            <motion.div
              variants={containerVariants}
              initial="hidden"
              animate={controls}
            >
              <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 gap-4">
                <div>
                  <h3 className="text-xl font-semibold text-gray-800 m-0">Community Issues</h3>
                  <p className="text-gray-600">Showing {filteredIssues.length} of {issues.length} issues</p>
                </div>
                
                <button className="flex items-center gap-2 px-4 py-2 bg-indigo-500 text-white rounded-lg text-sm font-medium hover:bg-indigo-600 transition">
                  <FiPlus />
                  <span>Add New Issue</span>
                </button>
              </div>
              
              {filteredIssues.length > 0 ? (
                viewMode === 'cards' ? (
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                    {filteredIssues.map((issue, index) => (
                      <IssueCard
                        key={issue.id}
                        issue={issue}
                        index={index}
                        expanded={expandedIssue === issue.id}
                        toggleExpand={() => toggleExpandIssue(issue.id)}
                      />
                    ))}
                  </div>
                ) : (
                  <div className="bg-white rounded-xl shadow-md overflow-hidden">
                    <table className="w-full border-collapse">
                      <thead>
                        <tr className="border-b border-gray-200">
                          <th className="p-4 text-left font-semibold text-gray-600">ID</th>
                          <th className="p-4 text-left font-semibold text-gray-600">Title</th>
                          <th className="p-4 text-left font-semibold text-gray-600">Priority</th>
                          <th className="p-4 text-left font-semibold text-gray-600">Status</th>
                          <th className="p-4 text-left font-semibold text-gray-600">Location</th>
                          <th className="p-4 text-left font-semibold text-gray-600">Date</th>
                          <th className="p-4 text-left font-semibold text-gray-600">Actions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {filteredIssues.map((issue) => (
                          <tr key={issue.id} className="border-b border-gray-200 last:border-b-0">
                            <td className="p-4">#{issue.id}</td>
                            <td className="p-4">{issue.title}</td>
                            <td className="p-4">
                              <span className={`inline-block px-3 py-1 rounded-full text-xs font-medium ${getPriorityColor(issue.priority)}`}>
                                {issue.priority}
                              </span>
                            </td>
                            <td className="p-4">
                              <span className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium ${getStatusColor(issue.status)}`}>
                                {getStatusIcon(issue.status)}
                                {issue.status}
                              </span>
                            </td>
                            <td className="p-4">{issue.location}</td>
                            <td className="p-4">{new Date(issue.date).toLocaleDateString()}</td>
                            <td className="p-4">
                              <div className="flex gap-2">
                                <button className="w-8 h-8 rounded-md bg-gray-50 border border-gray-200 flex items-center justify-center hover:bg-gray-100 transition">
                                  <FiEye />
                                </button>
                                <button className="w-8 h-8 rounded-md bg-gray-50 border border-gray-200 flex items-center justify-center hover:bg-gray-100 transition">
                                  <FiEdit />
                                </button>
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )
              ) : (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex flex-col items-center justify-center p-12 bg-white rounded-xl shadow-md text-center"
                >
                  <div className="w-16 h-16 rounded-full bg-gray-100 text-gray-400 flex items-center justify-center text-2xl mb-4">
                    <FiFilter />
                  </div>
                  <h5 className="text-lg font-semibold text-gray-800 mb-2">No issues found</h5>
                  <p className="text-gray-600 mb-6">Try adjusting your search or filters</p>
                  <button
                    onClick={() => {
                      setSearchTerm('');
                      setFilters({
                        status: 'all',
                        priority: 'all',
                        sort: 'newest'
                      });
                    }}
                    className="px-4 py-2 bg-indigo-500 text-white rounded-lg text-sm font-medium hover:bg-indigo-600 transition"
                  >
                    Reset Filters
                  </button>
                </motion.div>
              )}
            </motion.div>
          )}
        </motion.div>
      </div>
    </motion.div>
  );
}

// Helper functions
function getPriorityColor(priority) {
  switch (priority?.toLowerCase()) {
    case 'high': return 'bg-red-100 text-red-800 border-red-200';
    case 'medium': return 'bg-orange-100 text-orange-800 border-orange-200';
    case 'low': return 'bg-green-100 text-green-800 border-green-200';
    default: return 'bg-gray-100 text-gray-800 border-gray-200';
  }
}

function getStatusColor(status) {
  switch (status?.toLowerCase()) {
    case 'resolved': return 'bg-green-100 text-green-800 border-green-200';
    case 'in-progress': return 'bg-blue-100 text-blue-800 border-blue-200';
    case 'rejected': return 'bg-red-100 text-red-800 border-red-200';
    default: return 'bg-yellow-100 text-yellow-800 border-yellow-200';
  }
}

function getStatusIcon(status) {
  switch (status?.toLowerCase()) {
    case 'resolved': return <FiCheckCircle className="text-green-500" />;
    case 'in-progress': return <FiClock className="text-blue-500" />;
    case 'rejected': return <FiX className="text-red-500" />;
    default: return <FiAlertCircle className="text-yellow-500" />;
  }
}

function formatTimeAgo(dateString) {
  const date = new Date(dateString);
  const now = new Date();
  const diffInSeconds = Math.floor((now - date) / 1000);
  
  if (diffInSeconds < 60) return 'just now';
  if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)} minutes ago`;
  if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)} hours ago`;
  return `${Math.floor(diffInSeconds / 86400)} days ago`;
}

export default ViewIssues;