import React, { useState } from 'react';

// ============================================================================
// SVG Icon Components
// ============================================================================

const InnovateCorpLogo = () => (
  <svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M14 0L24.25 5.25V15.75L14 21L3.75 15.75V5.25L14 0Z" fill="#2563EB" />
    <path d="M14 28L28 20.5V7.5L14 15V28Z" fill="#60A5FA" />
    <path d="M0 7.5L14 15V28L0 20.5V7.5Z" fill="#3B82F6" />
  </svg>
);
const DocumentIcon = () => (
  <svg className="w-8 h-8 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
  </svg>
);
const TeamIcon = () => (
  <svg className="w-8 h-8 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.653-.124-1.282-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.653.124-1.282.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
  </svg>
);
const LightbulbIcon = () => (
  <svg className="w-8 h-8 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
  </svg>
);
const OfficeIcon = () => (
  <svg className="w-5 h-5 text-gray-500 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
  </svg>
);
const ChevronDownIcon = () => (
  <svg className="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" /></svg>
);
const PlayIcon = () => (
  <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" /></svg>
);
const CheckCircleIcon = () => (
  <svg className="w-6 h-6 text-blue-500" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" /></svg>
);


// ============================================================================
// Reusable UI Components
// ============================================================================

/**
 * A reusable button component with a primary style.
 * @param {object} props - The component props.
 * @param {React.ReactNode} props.children - The content of the button.
 * @param {function} props.onClick - The function to call when the button is clicked.
 * @param {string} [props.className] - Additional classes for styling.
 */
const Button = ({ children, onClick, className = '' }) => (
  <button
    onClick={onClick}
    className={`w-full bg-blue-600 text-white font-semibold py-2.5 rounded-lg hover:bg-blue-700 transition-colors ${className}`}
  >
    {children}
  </button>
);

/**
 * A progress bar component.
 * @param {object} props - The component props.
 * @param {number} props.value - The progress value (0-100).
 */
const ProgressBar = ({ value = 0 }) => (
  <div>
    <div className="flex justify-between items-center mb-1">
      <span className="text-sm text-gray-500">Progress</span>
      <span className="text-sm font-semibold text-blue-600">{value}%</span>
    </div>
    <div className="bg-gray-200 rounded-full h-2">
      <div className="bg-blue-600 h-2 rounded-full" style={{ width: `${value}%` }}></div>
    </div>
  </div>
);

/**
 * A component to display a stack of overlapping avatars.
 * @param {object} props - The component props.
 * @param {Array<{src: string, alt: string}>} props.avatars - An array of avatar objects.
 */
const AvatarStack = ({ avatars = [] }) => (
  <div className="flex -space-x-2">
    {avatars.map((avatar, index) => (
      <img
        key={index}
        className="inline-block h-8 w-8 rounded-full ring-2 ring-white"
        src={avatar.src}
        alt={avatar.alt}
      />
    ))}
  </div>
);

/**
 * A generic card component for displaying content sections.
 * @param {object} props - The component props.
 * @param {React.ReactNode} props.icon - The icon to display at the top of the card.
 * @param {string} props.title - The title of the card.
 * @param {string} props.description - The description text below the title.
 * @param {React.ReactNode} props.children - The main content of the card.
 * @param {string} props.buttonText - The text for the action button.
 * @param {function} props.onButtonClick - The handler for the button click.
 */
const ActionCard = ({ icon, title, description, children, buttonText, onButtonClick }) => (
  <div className="bg-white rounded-xl p-6 shadow-[0_4px_12px_rgba(0,0,0,0.05)] border border-gray-100 flex flex-col justify-between">
    <div>
      <div className="bg-blue-100 rounded-lg w-12 h-12 flex items-center justify-center">
        {icon}
      </div>
      <h3 className="font-bold text-gray-800 mt-4 text-lg">{title}</h3>
      <p className="text-sm text-gray-500 mt-1">{description}</p>
      <div className="mt-4">{children}</div>
    </div>
    <div className="mt-6">
      <Button onClick={onButtonClick}>{buttonText}</Button>
    </div>
  </div>
);

/**
 * A single item in the checklist.
 * @param {object} props - The component props.
 * @param {string} props.text - The text for the checklist item.
 * @param {boolean} props.isCompleted - Whether the item is completed.
 */
const ChecklistItem = ({ text, isCompleted }) => (
  <li className="flex items-center justify-between text-sm">
    <div className="flex items-center">
      <div className="w-4 h-4 border-2 border-gray-300 rounded-sm mr-3 flex-shrink-0"></div>
      <span className={`text-gray-700 ${isCompleted ? 'line-through' : ''}`}>{text}</span>
    </div>
    {isCompleted && <CheckCircleIcon />}
  </li>
);

/**
 * A generic widget for the sidebar.
 * @param {object} props - The component props.
 * @param {string} props.title - The title of the widget.
 * @param {React.ReactNode} [props.icon] - An optional icon to display next to the title.
 * @param {React.ReactNode} props.children - The content of the widget.
 */
const SidebarWidget = ({ title, icon, children }) => (
  <div className="bg-gray-50 rounded-xl p-5">
    <div className="flex items-center">
      {icon}
      <h3 className="font-bold text-gray-800">{title}</h3>
    </div>
    <div className="mt-4">{children}</div>
  </div>
);


// ============================================================================
// Section-Specific Components
// ============================================================================

/**
 * The header component for the dashboard.
 */
const DashboardHeader = () => {
  const navLinks = [
    { href: '#', text: 'Home', active: true },
    { href: '#', text: 'My Tasks' },
    { href: '#', text: 'Knowledge Base' },
    { href: '#', text: 'Team' },
  ];

  return (
    <header className="flex items-center justify-between pb-6 border-b border-gray-200">
      <div className="flex items-center gap-3">
        <InnovateCorpLogo />
        <span className="text-xl font-bold text-gray-800">InnovateCorp</span>
      </div>
      <nav className="hidden md:flex items-center gap-2">
        {navLinks.map(link => (
          <a
            key={link.text}
            href={link.href}
            className={`py-2 px-4 text-sm font-medium transition-colors ${
              link.active
                ? 'text-blue-600 font-semibold border-b-2 border-blue-600'
                : 'text-gray-500 hover:text-blue-600'
            }`}
          >
            {link.text}
          </a>
        ))}
      </nav>
      <div className="flex items-center gap-3">
        <img className="w-10 h-10 rounded-full object-cover" src="https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e?q=80&w=1887&auto=format&fit=crop" alt="User Avatar" />
        <button>
          <ChevronDownIcon />
        </button>
      </div>
    </header>
  );
};

/**
 * The welcome banner component.
 * @param {object} props - The component props.
 * @param {string} props.userName - The name of the user to welcome.
 * @param {string} props.subtitle - The subtitle text below the user's name.
 */
const WelcomeBanner = ({ userName, subtitle }) => (
  <div className="bg-gray-50 rounded-xl p-8 flex flex-col md:flex-row justify-between items-center gap-8">
    <div className="text-center md:text-left">
      <h1 className="text-4xl font-bold text-blue-600">Welcome</h1>
      <h1 className="text-4xl font-bold text-gray-800 mt-1">{userName}</h1>
      <p className="text-gray-500 mt-3">{subtitle}</p>
    </div>
    <div className="flex-shrink-0">
        <div className="relative">
            <img src="https://i.imgur.com/gPMARe6.png" alt="Company Overview Thumbnail" className="w-64 h-40 object-cover rounded-lg" />
            <button className="absolute inset-0 w-full h-full flex items-center justify-center">
                <div className="w-14 h-14 bg-white/30 backdrop-blur-sm rounded-full flex items-center justify-center">
                    <PlayIcon />
                </div>
            </button>
        </div>
        <p className="text-center text-sm text-gray-600 mt-2">Company Overview</p>
    </div>
  </div>
);

/**
 * The main component for the onboarding dashboard.
 */
const OnboardingDashboard = () => {
  const [checklistItems, setChecklistItems] = useState([
    { id: 1, text: 'Setup laptop', isCompleted: true },
    { id: 2, text: 'Attend orientation', isCompleted: false },
    { id: 3, text: 'Meet your buddy', isCompleted: true },
    { id: 4, text: 'Complete HR forms', isCompleted: true },
  ]);

  const teamAvatars = [
    { src: "https://images.unsplash.com/photo-1491528323818-fdd1faba62cc?q=80&w=2070&auto=format&fit=crop", alt: "User 1" },
    { src: "https://images.unsplash.com/photo-1552058544-f2b08422138a?q=80&w=1899&auto=format&fit=crop", alt: "User 2" },
    { src: "https://images.unsplash.com/photo-1580489944761-15a19d654956?q=80&w=1961&auto=format&fit=crop", alt: "User 3" },
    { src: "https://images.unsplash.com/photo-1544005313-94ddf0286df2?q=80&w=1888&auto=format&fit=crop", alt: "User 4" },
    { src: "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?q=80&w=1887&auto=format&fit=crop", alt: "User 5" },
  ];

  return (
    <div className="bg-slate-200 min-h-screen font-sans flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-xl w-full max-w-7xl p-6 lg:p-8">
        
        <DashboardHeader />

        <main className="mt-8">
          <WelcomeBanner 
            userName="Alex Chen" 
            subtitle="First day agenda | Start Date: June 24, 2024" 
          />

          <div className="mt-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 grid grid-cols-1 md:grid-cols-3 gap-6">
              
              <ActionCard
                icon={<DocumentIcon />}
                title="Complete Your Forms"
                description="3 of 5 forms completed"
                buttonText="Continue Forms"
                onButtonClick={() => alert('Navigating to forms...')}
              >
                <ProgressBar value={60} />
              </ActionCard>

              <ActionCard
                icon={<TeamIcon />}
                title="Meet Your Team"
                description="Your mentor Sarah and 12 team members"
                buttonText="View Team"
                onButtonClick={() => alert('Navigating to team page...')}
              >
                <AvatarStack avatars={teamAvatars} />
              </ActionCard>
              
              <ActionCard
                icon={<LightbulbIcon />}
                title="Knowledge Base"
                description="Quick answers to common questions"
                buttonText="Browse FAQS"
                onButtonClick={() => alert('Navigating to knowledge base...')}
              />

            </div>

            <aside className="space-y-6">
              
              <SidebarWidget title="First Day Checklist">
                <ul className="space-y-3">
                  {checklistItems.map(item => (
                    <ChecklistItem key={item.id} text={item.text} isCompleted={item.isCompleted} />
                  ))}
                </ul>
              </SidebarWidget>

              <SidebarWidget title="Office Location" icon={<OfficeIcon />}>
                <div className="flex items-start gap-4">
                  <img src="https://i.imgur.com/gKjym3f.png" alt="Map" className="w-16 h-16 rounded-lg object-cover" />
                  <div>
                    <p className="text-sm font-medium text-gray-800">123 Tech Drive, Suite 400.</p>
                    <p className="text-sm text-gray-500">Innovation City</p>
                  </div>
                </div>
              </SidebarWidget>

              <SidebarWidget title="Upcoming Schedule">
                <div>
                    <p className="text-sm font-semibold text-gray-800">9:00 AM - New Hire Orientation</p>
                    <p className="text-sm text-gray-500 mt-1">Team Meet & Greet</p>
                </div>
              </SidebarWidget>

            </aside>
          </div>
        </main>
      </div>
    </div>
  );
};

export default OnboardingDashboard;