import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Search, 
  Filter, 
  MapPin, 
  Calendar, 
  Eye, 
  Edit, 
  CheckCircle, 
  Clock, 
  XCircle, 
  AlertCircle,
  Inbox,
  ChevronDown,
  Grid,
  List,
  TrendingUp,
  BarChart3,
  RefreshCw,
  Download,
  MoreVertical,
  User,
  Tag
} from 'lucide-react';
import './StatusDisplay.css';

function StatusDisplay({ issues }) {
  const [filter, setFilter] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState('newest');
  const [viewMode, setViewMode] = useState('grid');
  const [isLoading, setIsLoading] = useState(true);
  const [filteredIssues, setFilteredIssues] = useState([]);
  
  // Simulate loading data
  useEffect(() => {
    setIsLoading(true);
    const timer = setTimeout(() => {
      applyFiltersAndSort();
      setIsLoading(false);
    }, 800);
    return () => clearTimeout(timer);
  }, [issues, filter, searchTerm, sortBy]);
  
  const applyFiltersAndSort = () => {
    let result = [...issues];
    
    // Apply filter
    if (filter !== 'all') {
      result = result.filter(issue => issue.status === filter);
    }
    
    // Apply search
    if (searchTerm) {
      const query = searchTerm.toLowerCase();
      result = result.filter(issue => 
        issue.id.toString().includes(query) || 
        (issue.address && issue.address.toLowerCase().includes(query)) ||
        (issue.issue_type && issue.issue_type.toLowerCase().includes(query))
      );
    }
    
    // Apply sorting
    result.sort((a, b) => {
      if (sortBy === 'newest') {
        return new Date(b.timestamp) - new Date(a.timestamp);
      } else if (sortBy === 'oldest') {
        return new Date(a.timestamp) - new Date(b.timestamp);
      } else if (sortBy === 'severity') {
        const severityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
        return (severityOrder[b.severity] || 0) - (severityOrder[a.severity] || 0);
      }
      return 0;
    });
    
    setFilteredIssues(result);
  };
  
  const statusCounts = issues.reduce((acc, issue) => {
    acc[issue.status] = (acc[issue.status] || 0) + 1;
    return acc;
  }, {});
  
  const statusIcons = {
    'pending': <Clock className="w-4 h-4" />,
    'in-progress': <AlertCircle className="w-4 h-4" />,
    'resolved': <CheckCircle className="w-4 h-4" />,
    'rejected': <XCircle className="w-4 h-4" />,
    'new': <Inbox className="w-4 h-4" />
  };
  
  const statusColors = {
    'pending': 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
    'in-progress': 'bg-blue-500/20 text-blue-400 border-blue-500/30',
    'resolved': 'bg-green-500/20 text-green-400 border-green-500/30',
    'rejected': 'bg-red-500/20 text-red-400 border-red-500/30',
    'new': 'bg-gray-500/20 text-gray-400 border-gray-500/30'
  };
  
  const severityColors = {
    'low': 'bg-green-500/20 text-green-400 border-green-500/30',
    'medium': 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
    'high': 'bg-orange-500/20 text-orange-400 border-orange-500/30',
    'critical': 'bg-red-500/20 text-red-400 border-red-500/30'
  };
  
  const severityIcons = {
    'low': <CheckCircle className="w-4 h-4" />,
    'medium': <Clock className="w-4 h-4" />,
    'high': <AlertCircle className="w-4 h-4" />,
    'critical': <XCircle className="w-4 h-4" />
  };
  
  const statusOptions = [
    { value: 'all', label: 'All Issues', icon: <Inbox className="w-4 h-4" /> },
    { value: 'new', label: 'New', icon: <Inbox className="w-4 h-4" /> },
    { value: 'pending', label: 'Pending', icon: <Clock className="w-4 h-4" /> },
    { value: 'in-progress', label: 'In Progress', icon: <AlertCircle className="w-4 h-4" /> },
    { value: 'resolved', label: 'Resolved', icon: <CheckCircle className="w-4 h-4" /> },
    { value: 'rejected', label: 'Rejected', icon: <XCircle className="w-4 h-4" /> }
  ];
  
  const sortOptions = [
    { value: 'newest', label: 'Newest First' },
    { value: 'oldest', label: 'Oldest First' },
    { value: 'severity', label: 'By Severity' }
  ];
  
  // Skeleton loader for issues
  const SkeletonCard = () => (
    <motion.div 
      className="issue-card skeleton"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <div className="card-header">
        <div className="skeleton skeleton-id"></div>
        <div className="skeleton skeleton-status"></div>
      </div>
      <div className="card-body">
        <div className="skeleton skeleton-type"></div>
        <div className="skeleton skeleton-severity"></div>
        <div className="skeleton skeleton-address"></div>
        <div className="skeleton skeleton-date"></div>
      </div>
      <div className="card-footer">
        <div className="skeleton skeleton-button"></div>
        <div className="skeleton skeleton-button"></div>
      </div>
    </motion.div>
  );
  
  return (
    <motion.div 
      className="status-display-container"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      {/* Header Section */}
      <motion.div 
        className="status-header"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="header-left">
          <div className="header-title-container">
            <div className="header-icon-bg">
              <TrendingUp className="header-icon" />
            </div>
            <div>
              <h3 className="header-title">Submitted Issues</h3>
              <p className="header-subtitle">Track and manage all reported issues</p>
            </div>
          </div>
          
          <div className="stats-container">
            <div className="stat-item">
              <div className="stat-value">{issues.length}</div>
              <div className="stat-label">Total</div>
            </div>
            <div className="stat-item">
              <div className="stat-value">{statusCounts['resolved'] || 0}</div>
              <div className="stat-label">Resolved</div>
            </div>
            <div className="stat-item">
              <div className="stat-value">{statusCounts['in-progress'] || 0}</div>
              <div className="stat-label">In Progress</div>
            </div>
          </div>
        </div>
        
        <div className="header-actions">
          <motion.button 
            className="icon-button"
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
          >
            <RefreshCw className="w-5 h-5" />
          </motion.button>
          
          <motion.button 
            className="icon-button"
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
          >
            <Download className="w-5 h-5" />
          </motion.button>
        </div>
      </motion.div>
      
      {/* Controls Section */}
      <motion.div 
        className="controls-container"
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        <div className="search-container">
          <Search className="search-icon" />
          <input
            type="text"
            placeholder="Search issues by ID, type, or location..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="search-input"
          />
        </div>
        
        <div className="filter-controls">
          <div className="filter-dropdown">
            <Filter className="filter-icon" />
            <select 
              value={filter} 
              onChange={(e) => setFilter(e.target.value)}
              className="filter-select"
            >
              {statusOptions.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
          
          <div className="sort-dropdown">
            <select 
              value={sortBy} 
              onChange={(e) => setSortBy(e.target.value)}
              className="sort-select"
            >
              {sortOptions.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
          
          <div className="view-toggle">
            <motion.button 
              className={`view-button ${viewMode === 'grid' ? 'active' : ''}`}
              onClick={() => setViewMode('grid')}
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
            >
              <Grid className="w-4 h-4" />
            </motion.button>
            <motion.button 
              className={`view-button ${viewMode === 'list' ? 'active' : ''}`}
              onClick={() => setViewMode('list')}
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
            >
              <List className="w-4 h-4" />
            </motion.button>
          </div>
        </div>
      </motion.div>
      
      {/* Status Pills */}
      <motion.div 
        className="status-pills-container"
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.3 }}
      >
        {statusOptions.map((option) => (
          <motion.button
            key={option.value}
            className={`status-pill ${filter === option.value ? 'active' : ''}`}
            onClick={() => setFilter(option.value)}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <div className="pill-icon">
              {option.icon}
            </div>
            <span className="pill-label">{option.label}</span>
            <span className="pill-count">
              {option.value === 'all' ? issues.length : (statusCounts[option.value] || 0)}
            </span>
          </motion.button>
        ))}
      </motion.div>
      
      {/* Issues Content */}
      {isLoading ? (
        <div className={`issues-container ${viewMode}`}>
          {Array.from({ length: 6 }).map((_, index) => (
            <SkeletonCard key={index} />
          ))}
        </div>
      ) : filteredIssues.length === 0 ? (
        <motion.div 
          className="empty-state"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <motion.div 
            className="empty-icon"
            animate={{ 
              y: [0, -10, 0],
              rotate: [0, 5, 0, -5, 0]
            }}
            transition={{ 
              duration: 4, 
              repeat: Infinity,
              ease: "easeInOut" 
            }}
          >
            <Inbox className="w-16 h-16 text-gray-400" />
          </motion.div>
          <h4 className="empty-title">No issues found</h4>
          <p className="empty-description">
            {searchTerm ? 'Try a different search term' : 'No issues match the current filter'}
          </p>
          <motion.button 
            className="empty-cta"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => {
              setSearchTerm('');
              setFilter('all');
            }}
          >
            Clear Filters
          </motion.button>
        </motion.div>
      ) : (
        <div className={`issues-container ${viewMode}`}>
          <AnimatePresence>
            {filteredIssues.map((issue) => (
              <motion.div
                key={issue.id}
                className="issue-card"
                layout
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.5 }}
                whileHover={{ y: -5 }}
              >
                <div className="card-header">
                  <div className="issue-id-container">
                    <span className="issue-id">#{issue.id}</span>
                    <div className={`status-badge ${statusColors[issue.status] || 'bg-gray-500/20 text-gray-400 border-gray-500/30'}`}>
                      {statusIcons[issue.status]}
                      <span>{issue.status.replace('-', ' ')}</span>
                    </div>
                  </div>
                  
                  <motion.button 
                    className="icon-button"
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                  >
                    <MoreVertical className="w-4 h-4" />
                  </motion.button>
                </div>
                
                <div className="card-body">
                  <div className="issue-row">
                    <div className="issue-property">
                      <div className="property-label">
                        <Tag className="w-4 h-4" />
                        <span>Type</span>
                      </div>
                      <span className="issue-type">{issue.issue_type || 'Not specified'}</span>
                    </div>
                    
                    <div className="issue-property">
                      <div className="property-label">
                        <AlertCircle className="w-4 h-4" />
                        <span>Severity</span>
                      </div>
                      <div className={`severity-badge ${severityColors[issue.severity?.toLowerCase()] || 'bg-gray-500/20 text-gray-400 border-gray-500/30'}`}>
                        {severityIcons[issue.severity?.toLowerCase()]}
                        <span>{issue.severity || 'Not specified'}</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="issue-row">
                    <div className="issue-property">
                      <div className="property-label">
                        <MapPin className="w-4 h-4" />
                        <span>Location</span>
                      </div>
                      <span className="issue-address">
                        {issue.address || 'No address provided'}
                      </span>
                    </div>
                  </div>
                  
                  {issue.timestamp && (
                    <div className="issue-row">
                      <div className="issue-property">
                        <div className="property-label">
                          <Calendar className="w-4 h-4" />
                          <span>Reported</span>
                        </div>
                        <span className="issue-date">
                          {new Date(issue.timestamp).toLocaleString()}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
                
                <div className="card-footer">
                  <motion.button 
                    className="action-btn view-btn"
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <Eye className="w-4 h-4" />
                    <span>View</span>
                  </motion.button>
                  <motion.button 
                    className="action-btn update-btn"
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <Edit className="w-4 h-4" />
                    <span>Update</span>
                  </motion.button>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      )}
    </motion.div>
  );
}

export default StatusDisplay;