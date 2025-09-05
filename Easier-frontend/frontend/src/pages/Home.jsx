import React, { useEffect, useState, useMemo } from 'react';
import { motion, useScroll, useTransform, AnimatePresence } from 'framer-motion';
import { useInView } from 'react-intersection-observer';
import { 
  FiArrowRight, FiZap, FiShield, FiClock, FiUsers, FiTrendingUp, 
  FiAward, FiStar, FiCheckCircle, FiPlay, FiArrowDown, FiUpload, 
  FiMapPin, FiFileText, FiTarget, FiTrendingDown
} from 'react-icons/fi';
import { Link } from 'react-router-dom';

const FeatureCard = React.forwardRef(({ icon, title, description, delay }, ref) => {
  const [inViewRef, inView] = useInView({ threshold: 0.1 });
  
  return (
    <motion.div
      ref={inViewRef}
      className="w-full"
      style={{
        opacity: inView ? 1 : 0,
        transform: inView ? 'translateY(0)' : 'translateY(20px)',
        transition: `all 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275) ${delay * 0.2}s`
      }}
    >
      <div className="bg-gray-900/80 backdrop-blur-sm border border-gray-800 p-10 rounded-3xl shadow-xl transition-all duration-300 ease-in-out hover:-translate-y-2 hover:shadow-2xl hover:shadow-blue-500/20 h-full flex flex-col items-center text-center relative overflow-hidden group">
        <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500 to-blue-700"></div>
        <div className="flex justify-center items-center w-24 h-24 bg-gradient-to-br from-blue-500/20 to-blue-700/20 rounded-2xl mb-7 text-blue-400 text-4xl relative overflow-hidden group-hover:after:opacity-20 after:content-[''] after:absolute after:top-0 after:left-0 after:w-full after:h-full after:rounded-2xl after:bg-gradient-to-br after:from-blue-500 after:to-blue-700 after:opacity-0 after:transition-opacity after:duration-300 after:ease-in-out after:-z-10">
          {icon}
        </div>
        <h5 className="text-2xl font-bold text-white mb-4">{title}</h5>
        <p className="text-gray-400 leading-relaxed text-lg">{description}</p>
      </div>
    </motion.div>
  );
});

const StatCard = React.forwardRef(({ icon, number, label, suffix = '', delay }, ref) => {
  const [inViewRef, inView] = useInView({ threshold: 0.1 });
  const [count, setCount] = useState(0);

  useEffect(() => {
    if (inView) {
      const target = parseInt(number.replace(/\D/g, ''), 10);
      const increment = target / 50;
      const timer = setInterval(() => {
        setCount(prev => {
          if (prev < target) {
            return Math.ceil(prev + increment);
          } else {
            clearInterval(timer);
            return target;
          }
        });
      }, 30);
      return () => clearInterval(timer);
    }
  }, [inView, number]);
  
  return (
    <motion.div
      ref={inViewRef}
      className="text-center transition-transform duration-300 ease-in-out hover:-translate-y-2 w-full"
      style={{
        opacity: inView ? 1 : 0,
        transform: inView ? 'translateY(0)' : 'translateY(20px)',
        transition: `all 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275) ${delay * 0.2}s`
      }}
    >
      <div className="flex justify-center items-center w-[70px] h-[70px] bg-white/10 backdrop-blur-xl rounded-2xl mx-auto mb-6 text-white text-3xl">{icon}</div>
      <h3 className="text-6xl font-extrabold mb-2 leading-none bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-300">{count}{suffix}</h3>
      <p className="text-xl text-gray-300 opacity-90 font-light m-0">{label}</p>
    </motion.div>
  );
});

const TestimonialCard = React.forwardRef(({ name, role, content, avatar, delay }, ref) => {
  const [inViewRef, inView] = useInView({ threshold: 0.1 });
  
  return (
    <motion.div
      ref={inViewRef}
      className="w-full"
      style={{
        opacity: inView ? 1 : 0,
        transform: inView ? 'translateY(0)' : 'translateY(20px)',
        transition: `all 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275) ${delay * 0.2}s`
      }}
    >
      <div className="bg-gray-900/80 backdrop-blur-sm border border-gray-800 rounded-3xl shadow-xl transition-all duration-300 ease-in-out hover:-translate-y-2 hover:shadow-2xl hover:shadow-blue-500/20 h-full overflow-hidden">
        <div className="p-8">
          <p className="text-lg leading-relaxed text-gray-300 mb-6 italic">"{content}"</p>
          <div className="flex items-center">
            <div className="w-[50px] h-[50px] rounded-full flex items-center justify-center text-xl bg-gradient-to-br from-blue-500 to-blue-700 text-white mr-4">{avatar}</div>
            <div>
              <h4 className="text-lg font-bold text-white mb-0.5">{name}</h4>
              <p className="text-sm text-gray-400 m-0">{role}</p>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
});

function Home() {
  const [scrollY, setScrollY] = useState(0);
  const [isLoaded, setIsLoaded] = useState(false);
  
  useEffect(() => {
    const handleScroll = () => setScrollY(window.scrollY);
    window.addEventListener('scroll', handleScroll, { passive: true });
    
    const timer = setTimeout(() => setIsLoaded(true), 500);
    
    return () => {
      window.removeEventListener('scroll', handleScroll);
      clearTimeout(timer);
    };
  }, []);
  
  const floatingElements = useMemo(() => {
    return [...Array(12)].map((_, i) => ({
      left: Math.random() * 100,
      top: Math.random() * 100,
      width: 10 + Math.random() * 30,
      height: 10 + Math.random() * 30,
      opacity: 0.03 + Math.random() * 0.1,
      animationDelay: Math.random() * 5,
      animationDuration: 8 + Math.random() * 6,
      blur: Math.random() > 0.5 ? 'blur(1px)' : 'none',
    }));
  }, []);
  
  const testimonials = [
    { name: "Sarah Johnson", role: "City Council Member", content: "SnapFix AI has revolutionized how we handle citizen reports. The AI classification saves us countless hours of manual sorting.", avatar: "👩‍💼" },
    { name: "Michael Chen", role: "Small Business Owner", content: "When my storefront was damaged, SnapFix AI helped me report it instantly. The issue was resolved within 24 hours!", avatar: "👨‍💼" },
    { name: "Emma Rodriguez", role: "Community Volunteer", content: "I love how easy it is to report issues in my neighborhood. SnapFix AI makes community engagement effortless.", avatar: "👩‍🦱" }
  ];
  
  return (
    <div className={`font-sans antialiased bg-black ${isLoaded ? 'opacity-100' : 'opacity-0'} transition-opacity duration-500 ease-in`}>
      {/* Hero Section with Parallax Effect */}
      <section className="relative overflow-hidden">
        <div 
          className="w-full bg-gradient-to-br from-gray-900 via-black to-gray-900 min-h-screen flex items-center justify-center overflow-hidden relative" 
          style={{ transform: `translateY(${scrollY * 0.4}px)` }}
        >
          {/* Animated floating elements */}
          <div className="absolute inset-0">
            {floatingElements.map((item, i) => (
              <div
                key={i}
                className="absolute bg-blue-500/20 rounded-full animate-float-slow pointer-events-none"
                style={{
                  left: `${item.left}%`,
                  top: `${item.top}%`,
                  width: `${item.width}px`,
                  height: `${item.height}px`,
                  opacity: item.opacity,
                  animationDelay: `${item.animationDelay}s`,
                  animationDuration: `${item.animationDuration}s`,
                  filter: item.blur,
                }}
              />
            ))}
          </div>
          
          {/* Animated grid overlay */}
          <div className="absolute inset-0 z-10 bg-grid-overlay"></div>
          
          <div className="hero-content relative z-20 text-center text-white max-w-5xl px-5">
            <motion.div className="text-left max-w-3xl">
  <h1 className="!text-4xl md:!text-4xl lg:!text-8xl font-extrabold mb-4 leading-tight tracking-tighter text-shadow-xl">
  <span className="bg-gradient-to-r from-blue-400 via-blue-500 to-blue-300 bg-clip-text text-transparent relative after:content-[''] after:absolute after:bottom-0 after:left-0 after:w-full after:h-2 after:bg-blue-500/30 after:rounded-md after:-z-10 inline-block">
    Eaiser
  </span> AI is here...
</h1>

  
  <p className="text-lg md:text-xl lg:text-2xl opacity-90 leading-relaxed font-light">
    Transform your community with intelligent issue reporting. Our AI analyzes and categorizes problems in real-time for faster resolution.
  </p>

  <div className="mt-6 flex gap-4 flex-wrap">
    <Link 
      to="/report" 
      className="flex items-center px-3 py-2 rounded-full text-lg font-semibold transition-all duration-300 ease-in-out bg-gradient-to-br from-blue-500 to-blue-700 text-white shadow-lg shadow-blue-500/40 hover:-translate-y-1 hover:shadow-xl hover:shadow-blue-500/50"
    >
      Report an Issue 
      <span className="ml-2 transition-transform duration-300 ease-in-out group-hover:translate-x-1">→</span>
    </Link>
  </div>
</motion.div>

          </div>
          
          {/* Scroll indicator */}
          <div className="absolute bottom-8 left-1/2 -translate-x-1/2 z-20">
            <div className="w-3 h-3 rounded-full bg-white/70 animate-pulse-custom"></div>
          </div>
        </div>
      </section>
      
      {/* Features Section */}
      <section className="py-24 bg-gradient-to-br from-gray-900 to-black relative">
        <div className="container mx-auto px-4 max-w-7xl">
          <div className="text-center mb-16">
            <motion.div className="inline-block bg-gradient-to-br from-blue-500 to-blue-700 text-white px-4 py-1.5 rounded-full font-semibold tracking-wide text-xs mb-4">CORE FEATURES</motion.div>
            <motion.h2 className="text-4xl lg:text-5xl font-extrabold mb-4 text-white">
              Why <span className="text-blue-400 relative after:content-[''] after:absolute after:bottom-0 after:left-0 after:w-full after:h-1.5 after:bg-blue-500/30 after:rounded-sm after:-z-10">Eaiser AI</span> Stands Apart
            </motion.h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              Our advanced AI technology streamlines the entire issue resolution process
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-10 mt-12">
            <FeatureCard
              icon={<FiUpload className="text-blue-400 text-4xl" />}
              title="Smart AI Analysis"
              description="Our neural networks instantly classify issues as public or business-related with 95% accuracy."
              delay={0}
            />
            <FeatureCard
              icon={<FiMapPin className="text-blue-400 text-4xl" />}
              title="Precision Location"
              description="GPS and visual data pinpoint exact locations for faster response times."
              delay={1}
            />
            <FeatureCard
              icon={<FiFileText className="text-blue-400 text-4xl" />}
              title="Automated Reporting"
              description="Generate professional reports automatically sent to the appropriate authorities."
              delay={2}
            />
          </div>
        </div>
      </section>
      
      {/* Stats Section */}
      <section className="py-24 bg-gradient-to-br from-gray-800 via-gray-900 to-black text-white relative overflow-hidden">
        <div className="absolute inset-0 z-0 bg-stats-pattern opacity-30"></div>
        <div className="container mx-auto px-4 max-w-7xl relative z-10">
          <div className="text-center mb-16">
            <motion.div className="inline-block bg-white/10 backdrop-blur-md px-4 py-1.5 rounded-full font-semibold tracking-wide text-xs mb-4">IMPACT METRICS</motion.div>
            <motion.h2 className="text-4xl lg:text-5xl font-extrabold mb-4 text-white">
              Making a <span className="relative after:content-[''] after:absolute after:bottom-0 after:left-0 after:w-full after:h-1.5 after:bg-blue-500/40 after:rounded-sm after:-z-10">Measurable Difference</span>
            </motion.h2>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-12 relative z-10">
            <StatCard
              icon={<FiCheckCircle className="text-white text-3xl" />}
              number="12K+"
              label="Issues Resolved"
              delay={0}
            />
            <StatCard
              icon={<FiClock className="text-white text-3xl" />}
              number="18"
              label="Avg. Resolution (hours)"
              delay={1}
            />
            <StatCard
              icon={<FiUsers className="text-white text-3xl" />}
              number="95%"
              label="User Satisfaction"
              suffix="%"
              delay={2}
            />
          </div>
        </div>
      </section>
      
      {/* Testimonials Section */}
      <section className="py-24 bg-gradient-to-br from-black to-gray-900">
        <div className="container mx-auto px-4 max-w-7xl">
          <div className="text-center mb-16">
            <motion.div className="inline-block bg-gradient-to-br from-blue-500 to-blue-700 text-white px-4 py-1.5 rounded-full font-semibold tracking-wide text-xs mb-4">SUCCESS STORIES</motion.div>
            <motion.h2 className="text-4xl lg:text-5xl font-extrabold mb-4 text-white">
              What Our <span className="text-blue-400 relative after:content-[''] after:absolute after:bottom-0 after:left-0 after:w-full after:h-1.5 after:bg-blue-500/30 after:rounded-sm after:-z-10">Users Say</span>
            </motion.h2>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {testimonials.map((testimonial, index) => (
              <TestimonialCard
                key={index}
                name={testimonial.name}
                role={testimonial.role}
                content={testimonial.content}
                avatar={testimonial.avatar}
                delay={index}
              />
            ))}
          </div>
        </div>
      </section>
      
      {/* CTA Section */}
      <section className="py-24 bg-gradient-to-br from-gray-900 to-black relative overflow-hidden">
        <div className="absolute inset-0 z-0 bg-radial-gradient"></div>
        <div className="container mx-auto px-4 max-w-4xl relative z-10">
          <motion.div className="text-center text-white">
            <div className="inline-block bg-blue-500/20 text-blue-300 px-4 py-1.5 rounded-full font-semibold tracking-wide text-xs mb-4">JOIN THE MOVEMENT</div>
            <h2 className="text-4xl lg:text-5xl font-bold mb-4 text-white">Ready to Improve Your Community?</h2>
            <p className="text-xl text-gray-300 mb-10 leading-relaxed">
              Join thousands of users making a difference with AI-powered issue reporting
            </p>
            <div className="flex gap-4 justify-center flex-wrap">
              <Link to="/report" className="flex items-center px-8 py-4 rounded-full text-lg font-semibold transition-all duration-300 ease-in-out border-none cursor-pointer bg-gradient-to-br from-blue-500 to-blue-700 text-white shadow-lg shadow-blue-500/40 hover:-translate-y-1 hover:shadow-xl hover:shadow-blue-500/50">
                Get Started Now
              </Link>
              <Link to="/features" className="flex items-center px-8 py-4 rounded-full text-lg font-semibold transition-all duration-300 ease-in-out cursor-pointer bg-white/10 backdrop-blur-md text-white border border-white/20 hover:bg-white/20 hover:border-white/30 hover:-translate-y-1">
                Learn More
              </Link>
            </div>
          </motion.div>
        </div>
      </section>
      
      {/* Footer */}
      <footer className="py-12 bg-black text-gray-400 border-t border-gray-800">
        <div className="container mx-auto px-4 max-w-7xl">
          <div className="flex flex-col items-center text-center">
            <div className="text-3xl font-extrabold text-white mb-6">
              <span className="bg-gradient-to-r from-blue-500 to-blue-400 bg-clip-text text-transparent">Eaiser</span> AI
            </div>
            <div className="flex gap-6 mb-6 flex-wrap justify-center">
              <Link to="/about" className="text-gray-400 text-lg hover:text-blue-400 transition-colors duration-300">About</Link>
              <Link to="/features" className="text-gray-400 text-lg hover:text-blue-400 transition-colors duration-300">Features</Link>
              <Link to="/contact" className="text-gray-400 text-lg hover:text-blue-400 transition-colors duration-300">Contact</Link>
              <Link to="/privacy" className="text-gray-400 text-lg hover:text-blue-400 transition-colors duration-300">Privacy</Link>
            </div>
            <div className="text-sm opacity-70">
              © {new Date().getFullYear()} Easier AI. All rights reserved.
            </div>
          </div>
        </div>
      </footer>
      
      <style>{`
        @keyframes pulse-custom {
          0%, 100% { transform: scale(1); opacity: 0.7; }
          50% { transform: scale(1.5); opacity: 0.3; }
        }
        
        @keyframes float-slow {
          0%, 100% { transform: translateY(0) rotate(0deg); }
          50% { transform: translateY(-20px) rotate(5deg); }
        }
        
        .animate-float-slow { animation: float-slow var(--duration, 8s) ease-in-out infinite; }
        .animate-pulse-custom { animation: pulse-custom 2s infinite; }
        .text-shadow-xl { text-shadow: 0 4px 12px rgba(0,0,0,0.5); }
        
        .bg-grid-overlay {
          background-image: linear-gradient(rgba(59, 130, 246, 0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(59, 130, 246, 0.03) 1px, transparent 1px);
          background-size: 40px 40px;
        }

        .bg-stats-pattern {
          background-image: url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiPjxkZWZzPjxwYXR0ZXJuIGlkPSJwYXR0ZXJuIiB3aWR0aD0iNDAiIGhlaWdodD0iNDAiIHBhdHRlcm5Vbml0cz0idXNlclNwYWNlT25Vc2UiIHBhdHRlcm5UcmFuc2Zvcm09InJvdGF0ZSg0NSkiPjxyZWN0IHg9IjAiIHk9IjAiIHdpZHRoPSIyMCIgaGVpZ2h0PSIyMCIgZmlsbD0icmdiYSg1OSwgMTMwLCAyNDYsIDAuMDMpIi8+PC9wYXR0ZXJuPjwvZGVmcz48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSJ1cmwoI3BhdHRlcm4pIi8+PC9zdmc+');
        }
        
        .bg-radial-gradient {
          background: radial-gradient(circle at 80% 20%, rgba(59, 130, 246, 0.1), transparent 50%);
        }
      `}</style>
    </div>
  );
}

export default Home;