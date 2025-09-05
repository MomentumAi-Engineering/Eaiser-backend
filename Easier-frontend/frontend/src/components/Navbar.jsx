import React, { useState, useEffect, useRef } from "react";
import { Link, useLocation } from "react-router-dom";
import { motion, AnimatePresence } from 'framer-motion';
import {
  FiHome, FiAlertCircle, FiList, FiUser, FiSettings, FiLogOut, FiBell, FiMenu, FiX, FiZap
} from "react-icons/fi";

import logo from "../assets/logo.png"; // keep your path

function Navbar() {
  const location = useLocation();
  const [activeDropdown, setActiveDropdown] = useState(null);
  const [notifications] = useState(3);
  const [scrolled, setScrolled] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const dropdownRef = useRef(null);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    window.addEventListener("scroll", onScroll);

    const onClickOutside = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setActiveDropdown(null);
      }
    };
    document.addEventListener("mousedown", onClickOutside);

    return () => {
      window.removeEventListener("scroll", onScroll);
      document.removeEventListener("mousedown", onClickOutside);
    };
  }, []);

  const toggleDropdown = (menu) =>
    setActiveDropdown(activeDropdown === menu ? null : menu);

  const navLinks = [
    { path: "/", label: "Home", icon: <FiHome className="h-5 w-5 " /> },
    { path: "/report", label: "Report Issue", icon: <FiAlertCircle className="h-5 w-5" /> },
    { path: "/view", label: "My Issues", icon: <FiList className="h-5 w-5" /> },
  ];

  const logoVariants = {
    initial: { scale: 1 },
    hover: { 
      scale: 1.05,
      transition: { type: "spring", stiffness: 400, damping: 10 }
    },
    tap: { scale: 0.95 }
  };

  const navItemVariants = {
    initial: { y: -20, opacity: 0 },
    animate: { y: 0, opacity: 1 },
    hover: { 
      y: -2,
      transition: { type: "spring", stiffness: 400, damping: 10 }
    }
  };

  const mobileMenuVariants = {
    closed: {
      opacity: 0,
      y: -20,
      transition: {
        duration: 0.2,
        staggerChildren: 0.05,
        staggerDirection: -1
      }
    },
    open: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.3,
        staggerChildren: 0.1,
        delayChildren: 0.1
      }
    }
  };

  const mobileItemVariants = {
    closed: { x: -20, opacity: 0 },
    open: { x: 0, opacity: 1 }
  };

  const userMenuItems = [
    { path: "/profile", label: "My Profile", icon: <FiUser /> },
    { path: "/settings", label: "Settings", icon: <FiSettings /> },
    { path: "/logout", label: "Logout", icon: <FiLogOut /> },
  ];

  return (
    <>
      <motion.nav
        className={[
          "fixed z-[1000] h-16 transition-all duration-700 ease-out",
          scrolled
            ? // SCROLLED: Advanced glassmorphism with water glass effect
              "top-3 left-1/2 -translate-x-1/2 w-[94%] max-w-7xl rounded-3xl " +
              "bg-gradient-to-r from-white/[0.08] via-white/[0.12] to-white/[0.08] " +
              "backdrop-blur-2xl border border-white/20 " +
              "shadow-[0_8px_32px_rgba(0,0,0,0.3),0_0_0_1px_rgba(255,255,255,0.1),inset_0_1px_0_rgba(255,255,255,0.2)] " +
              "before:absolute before:inset-0 before:rounded-3xl before:bg-gradient-to-r before:from-transparent before:via-white/5 before:to-transparent before:animate-pulse"
            : // TOP: Professional gradient with glass overlay
              "top-0 left-0 w-full border-b border-white/10 " +
              "bg-gradient-to-r from-slate-900/95 via-indigo-900/90 to-slate-900/95 " +
              "backdrop-blur-md shadow-lg shadow-black/25"
        ].join(" ")}
        ref={dropdownRef}
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        transition={{ type: "spring", stiffness: 100, damping: 20 }}
      >
        <div className="h-full px-3 md:px-6 flex items-center justify-between">
          {/* Logo with Advanced Glass Effect */}
          <motion.div 
            className="flex items-center gap-3 relative"
            variants={logoVariants}
            initial="initial"
            whileHover="hover"
            whileTap="tap"
          >
            <Link to="/" className="flex items-center gap-2 md:gap-3 no-underline hover:no-underline group">
              <motion.div
                className="relative p-1.5 md:p-2 rounded-xl md:rounded-2xl bg-gradient-to-br from-white/20 via-white/10 to-white/5 backdrop-blur-xl border border-white/30 shadow-[0_8px_32px_rgba(0,0,0,0.3),inset_0_1px_0_rgba(255,255,255,0.3)]"
                whileHover={{ 
                  scale: 1.05,
                  boxShadow: "0 12px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.4)"
                }}
                transition={{ duration: 0.3 }}
              >
                <motion.img
                  src={logo}
                  alt="SnapFix Logo"
                  className="h-6 w-6 md:h-8 md:w-8 rounded-lg filter drop-shadow-lg"
                  whileHover={{ rotate: 360, filter: "brightness(1.2) saturate(1.3)" }}
                  transition={{ duration: 0.6, ease: "easeInOut" }}
                />
                <div className="absolute inset-0 rounded-xl md:rounded-2xl bg-gradient-to-br from-indigo-400/20 to-purple-400/20 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
              </motion.div>
              <div className="flex items-baseline gap-1">
                <motion.span 
                   className="text-xl md:text-2xl font-bold bg-gradient-to-r from-white via-blue-100 to-white bg-clip-text text-transparent tracking-tight filter drop-shadow-sm"
                   initial={{ opacity: 0, x: -20 }}
                   animate={{ opacity: 1, x: 0 }}
                   transition={{ delay: 0.2 }}
                   whileHover={{ 
                     filter: "drop-shadow(0 0 8px rgba(255,255,255,0.5))",
                     scale: 1.02
                   }}
                 >
                   EasierAI
                 </motion.span>
              </div>
            </Link>
          </motion.div>


          {/* Advanced Desktop Navigation Links with Glassmorphism */}
          <ul className="hidden lg:flex items-center gap-1 xl:gap-2">
            {navLinks.map((link, index) => {
              const isActive = location.pathname === link.path;
              return (
                <motion.li 
                  key={link.path}
                  variants={navItemVariants}
                  initial="initial"
                  animate="animate"
                  whileHover="hover"
                  transition={{ delay: index * 0.1 }}
                  className="relative"
                >
                  <Link
                    to={link.path}
                    className={[
                      "relative flex items-center px-3 lg:px-6 py-2 lg:py-3 text-white text-xs lg:text-sm font-medium rounded-2xl",
                      "transition-all duration-500 no-underline hover:no-underline decoration-none",
                      "overflow-hidden group backdrop-blur-xl border border-white/20",
                      "hover:shadow-[0_8px_32px_rgba(0,0,0,0.3),inset_0_1px_0_rgba(255,255,255,0.3)]",
                      "hover:scale-105 hover:border-white/30",
                      isActive
                        ? "bg-gradient-to-r from-white/20 via-white/15 to-white/10 shadow-[0_8px_32px_rgba(0,0,0,0.4),inset_0_1px_0_rgba(255,255,255,0.4)] border-white/40"
                        : "bg-gradient-to-r from-white/5 via-white/8 to-white/5 hover:from-white/15 hover:via-white/20 hover:to-white/15"
                    ].join(" ")}
                  >
                    <span className="relative z-10 filter drop-shadow-sm">{link.icon}</span>
                    <span className="relative z-10 ml-2 filter drop-shadow-sm">{link.label}</span>
                    
                    {/* Animated background glow */}
                    <motion.div
                      className="absolute inset-0 bg-gradient-to-r from-indigo-400/20 via-purple-400/20 to-indigo-400/20 opacity-0 group-hover:opacity-100 rounded-2xl"
                      animate={{
                        backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"]
                      }}
                      transition={{
                        duration: 3,
                        repeat: Infinity,
                        ease: "linear"
                      }}
                    />
                    
                    {/* Active state indicator */}
                    {isActive && (
                      <>
                        <motion.div
                          className="absolute bottom-1 left-1/2 -translate-x-1/2 w-8 h-1 bg-gradient-to-r from-indigo-400 to-purple-400 rounded-full shadow-lg"
                          layoutId="activeIndicator"
                          initial={{ scaleX: 0 }}
                          animate={{ scaleX: 1 }}
                          transition={{ type: "spring", bounce: 0.3, duration: 0.8 }}
                        />
                        <motion.div
                          className="absolute inset-0 bg-gradient-to-r from-indigo-400/10 to-purple-400/10 rounded-2xl"
                          animate={{
                            opacity: [0.3, 0.6, 0.3]
                          }}
                          transition={{
                            duration: 2,
                            repeat: Infinity,
                            ease: "easeInOut"
                          }}
                        />
                      </>
                    )}
                    
                    {/* Hover ripple effect */}
                    <motion.div
                      className="absolute inset-0 bg-white/10 rounded-2xl opacity-0 group-hover:opacity-100"
                      whileHover={{
                        scale: [1, 1.05, 1],
                        opacity: [0, 0.3, 0]
                      }}
                      transition={{ duration: 0.6 }}
                    />
                  </Link>
                </motion.li>
              );
            })}
          </ul>

          {/* Advanced Mobile Menu Button with Glass Effect */}
          <motion.button
            className="lg:hidden relative p-2 md:p-3 rounded-xl md:rounded-2xl bg-gradient-to-br from-white/20 via-white/15 to-white/10 backdrop-blur-xl border border-white/30 shadow-[0_8px_32px_rgba(0,0,0,0.3),inset_0_1px_0_rgba(255,255,255,0.3)] hover:shadow-[0_12px_40px_rgba(0,0,0,0.4),inset_0_1px_0_rgba(255,255,255,0.4)] transition-all duration-300 group overflow-hidden"
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            whileHover={{ 
              scale: 1.05,
              boxShadow: "0 12px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.4)"
            }}
            whileTap={{ scale: 0.95 }}
          >
            {/* Animated background glow */}
            <motion.div
              className="absolute inset-0 bg-gradient-to-r from-indigo-400/20 via-purple-400/20 to-indigo-400/20 opacity-0 group-hover:opacity-100 rounded-2xl"
              animate={{
                backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"]
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                ease: "linear"
              }}
            />
            
            <AnimatePresence mode="wait">
              {isMobileMenuOpen ? (
                <motion.div
                  key="close"
                  className="relative z-10"
                  initial={{ rotate: -90, opacity: 0 }}
                  animate={{ rotate: 0, opacity: 1 }}
                  exit={{ rotate: 90, opacity: 0 }}
                  transition={{ duration: 0.4, ease: "easeInOut" }}
                >
                  <FiX className="h-5 w-5 md:h-6 md:w-6 text-white filter drop-shadow-lg" />
                </motion.div>
              ) : (
                <motion.div
                  key="menu"
                  className="relative z-10"
                  initial={{ rotate: 90, opacity: 0 }}
                  animate={{ rotate: 0, opacity: 1 }}
                  exit={{ rotate: -90, opacity: 0 }}
                  transition={{ duration: 0.4, ease: "easeInOut" }}
                >
                  <FiMenu className="h-5 w-5 md:h-6 md:w-6 text-white filter drop-shadow-lg" />
                </motion.div>
              )}
            </AnimatePresence>
            
            {/* Ripple effect on tap */}
            <motion.div
              className="absolute inset-0 bg-white/20 rounded-2xl opacity-0"
              whileTap={{
                scale: [1, 1.2, 1],
                opacity: [0, 0.5, 0]
              }}
              transition={{ duration: 0.3 }}
            />
          </motion.button>

          {/* Right actions */}
          <div className="flex items-center gap-4">
            {/* Notifications */}
            <div className="relative">
              <button
                className="p-2 rounded-md hover:bg-white/10 transition"
                onClick={() => toggleDropdown("notifications")}
                aria-label="Notifications"
              >
                <FiBell className="text-white text-xl" />
                {notifications > 0 && (
                  <span className="absolute -top-1 -right-1 bg-pink-500 text-white text-xs px-1.5 rounded-full">
                    {notifications}
                  </span>
                )}
              </button>

              {activeDropdown === "notifications" && (
                <div className="absolute right-0 mt-2 w-80 bg-slate-900/95 rounded-lg shadow-lg border border-white/10 overflow-hidden">
                  <div className="flex justify-between items-center px-4 py-2 border-b border-white/10">
                    <h3 className="text-white text-sm font-semibold">Notifications</h3>
                    <button className="text-indigo-300 text-xs hover:text-indigo-400">
                      Mark all as read
                    </button>
                  </div>
                  <div>
                    <div className="px-4 py-2 border-b border-white/10 hover:bg-white/5">
                      <p className="text-white text-sm">Your issue #123 has been resolved</p>
                      <span className="text-xs text-slate-400">2 hours ago</span>
                    </div>
                    <div className="px-4 py-2 border-b border-white/10 hover:bg-white/5">
                      <p className="text-white text-sm">New comment on your report</p>
                      <span className="text-xs text-slate-400">1 day ago</span>
                    </div>
                    <div className="px-4 py-2 hover:bg-white/5">
                      <p className="text-white text-sm">System maintenance scheduled</p>
                      <span className="text-xs text-slate-400">3 days ago</span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* User Menu */}
            <div className="relative">
              <button
                className="w-9 h-9 rounded-full bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center text-white font-semibold"
                onClick={() => toggleDropdown("user")}
                aria-label="User menu"
              >
                U
              </button>

              {activeDropdown === "user" && (
                <div className="absolute right-0 mt-2 w-56 bg-slate-900/95 rounded-lg shadow-lg border border-white/10">
                  <div className="flex items-center p-4 border-b border-white/10">
                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center text-white font-semibold">
                      U
                    </div>
                    <div className="ml-3">
                      <h4 className="text-white text-sm font-semibold">User Name</h4>
                      <p className="text-slate-400 text-xs">user@example.com</p>
                    </div>
                  </div>
                  <div className="flex flex-col">
                    {userMenuItems.map((item) => (
                      <Link
                        to={item.path}
                        key={item.path}
                        className="px-4 py-2 text-slate-300 text-sm hover:bg-white/5 hover:text-white no-underline hover:no-underline"
                      >
                        {item.icon && <span className="mr-2 inline-block">{item.icon}</span>}
                        {item.label}
                      </Link>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </motion.nav>

      {/* Advanced Mobile Navigation Menu with Enhanced Glassmorphism */}
      <AnimatePresence>
        {isMobileMenuOpen && (
          <motion.div
            className="fixed inset-0 z-[999] md:hidden"
            variants={mobileMenuVariants}
            initial="closed"
            animate="open"
            exit="closed"
          >
            {/* Enhanced Mobile Menu Backdrop with Blur */}
            <motion.div
              className="absolute inset-0 bg-gradient-to-br from-black/60 via-slate-900/50 to-black/60 backdrop-blur-md"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsMobileMenuOpen(false)}
            />
            
            {/* Advanced Mobile Menu Content with Water Glass Effect */}
            <motion.div
              className="absolute top-20 left-2 right-2 md:left-4 md:right-4 bg-gradient-to-br from-slate-900/80 via-slate-800/85 to-slate-900/80 backdrop-blur-2xl rounded-2xl md:rounded-3xl border border-white/20 shadow-[0_20px_60px_rgba(0,0,0,0.5),inset_0_1px_0_rgba(255,255,255,0.2)] overflow-hidden"
              initial={{ scale: 0.9, opacity: 0, y: -30, rotateX: -15 }}
              animate={{ scale: 1, opacity: 1, y: 0, rotateX: 0 }}
              exit={{ scale: 0.9, opacity: 0, y: -30, rotateX: -15 }}
              transition={{ type: "spring", stiffness: 300, damping: 30 }}
            >
              {/* Glass overlay effect */}
              <div className="absolute inset-0 bg-gradient-to-br from-white/10 via-transparent to-white/5 pointer-events-none" />
              
              <div className="p-4 md:p-6 space-y-2 md:space-y-3 relative z-10">
                {navLinks.map((link, index) => {
                  const isActive = location.pathname === link.path;
                  return (
                    <motion.div
                      key={link.path}
                      variants={mobileItemVariants}
                      custom={index}
                      initial={{ opacity: 0, x: -30, rotateY: -15 }}
                      animate={{ opacity: 1, x: 0, rotateY: 0 }}
                      transition={{ 
                        delay: index * 0.1,
                        duration: 0.4,
                        ease: "easeOut"
                      }}
                    >
                      <Link
                        to={link.path}
                        className={[
                          "flex items-center p-3 md:p-4 rounded-xl md:rounded-2xl text-white no-underline hover:no-underline",
                          "transition-all duration-300 group relative overflow-hidden backdrop-blur-sm",
                          "border hover:shadow-[0_8px_32px_rgba(0,0,0,0.3),inset_0_1px_0_rgba(255,255,255,0.3)]",
                          "hover:scale-[1.02] hover:border-white/30",
                          isActive
                            ? "bg-gradient-to-r from-white/20 via-white/15 to-white/10 shadow-[0_8px_32px_rgba(0,0,0,0.4),inset_0_1px_0_rgba(255,255,255,0.4)] border-white/40"
                            : "bg-gradient-to-r from-white/5 via-white/8 to-white/5 hover:from-white/15 hover:via-white/20 hover:to-white/15 border-white/10"
                        ].join(" ")}
                        onClick={() => setIsMobileMenuOpen(false)}
                      >
                        {/* Animated background glow */}
                        <motion.div
                          className="absolute inset-0 bg-gradient-to-r from-indigo-400/20 via-purple-400/20 to-indigo-400/20 opacity-0 group-hover:opacity-100 rounded-2xl"
                          animate={{
                            backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"]
                          }}
                          transition={{
                            duration: 3,
                            repeat: Infinity,
                            ease: "linear"
                          }}
                        />
                        
                        <span className="relative z-10 text-base md:text-lg filter drop-shadow-sm">{link.icon}</span>
                        <span className="relative z-10 ml-3 md:ml-4 text-sm md:text-base font-medium filter drop-shadow-sm">{link.label}</span>
                        
                        {/* Active state indicator */}
                        {isActive && (
                          <>
                            <motion.div
                              className="relative z-10 ml-auto w-2 h-2 bg-gradient-to-r from-indigo-400 to-purple-400 rounded-full shadow-lg"
                              initial={{ scale: 0 }}
                              animate={{ scale: 1 }}
                              transition={{ delay: 0.2 }}
                            />
                            <motion.div
                              className="absolute left-2 top-1/2 -translate-y-1/2 w-1 h-8 bg-gradient-to-b from-indigo-400 to-purple-400 rounded-full shadow-lg"
                              layoutId="mobileActiveIndicator"
                              transition={{ type: "spring", bounce: 0.3, duration: 0.8 }}
                            />
                            <motion.div
                              className="absolute inset-0 bg-gradient-to-r from-indigo-400/10 to-purple-400/10 rounded-2xl"
                              animate={{
                                opacity: [0.3, 0.6, 0.3]
                              }}
                              transition={{
                                duration: 2,
                                repeat: Infinity,
                                ease: "easeInOut"
                              }}
                            />
                          </>
                        )}
                        
                        {/* Hover ripple effect */}
                        <motion.div
                          className="absolute inset-0 bg-white/10 rounded-2xl opacity-0 group-hover:opacity-100"
                          whileHover={{
                            scale: [1, 1.05, 1],
                            opacity: [0, 0.3, 0]
                          }}
                          transition={{ duration: 0.6 }}
                        />
                      </Link>
                    </motion.div>
                  );
                })}
              </div>
              
              {/* Bottom glass reflection */}
              <div className="absolute bottom-0 left-0 right-0 h-20 bg-gradient-to-t from-white/5 to-transparent pointer-events-none" />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

export default Navbar;
