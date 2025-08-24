import React from 'react';

// Sidebar Component
const Sidebar = () => (
  <div className="w-20 bg-white border-r p-4 flex flex-col items-center">
    <div className="bg-blue-500 p-3 rounded-full mb-4">
      {/* Profile SVG or icon here */}
    </div>
    <nav className="flex flex-col items-center space-y-4 mt-4">
      <div className="w-8 h-8 bg-gray-300 rounded-full"></div>
      <div className="w-8 h-8 bg-gray-300 rounded-full"></div>
      <div className="w-8 h-8 bg-gray-300 rounded-full"></div>
    </nav>
  </div>
);

// Header Component
const Header = () => (
  <header className="flex items-center justify-between mb-8">
    <h1 className="text-3xl font-bold">Welcome, Sarah!</h1>
    <div className="bg-white p-2 rounded-full">
      {/* Settings SVG or icon here */}
    </div>
  </header>
);

// Onboarding Overview Component
const OnboardingOverview = () => (
  <div className="bg-white p-6 rounded-lg shadow-md w-1/2">
    <h2 className="text-lg font-bold mb-4">My Onboarding Overview</h2>
    <div className="flex items-center mb-4">
      <div className="w-16 h-16 border-4 border-blue-500 rounded-full flex items-center justify-center">
        <span className="text-xl">40%</span>
      </div>
    </div>
    <div>
      <h3 className="font-semibold mb-2">Today's Tasks</h3>
      <ul className="space-y-2">
        <TaskItem text="Complete Benefits Enrollment" />
        <TaskItem text="Review Employee Handbook" checked date="Jul 26" dateColor="bg-yellow-500" />
        <TaskItem text="Set Up Workstation" date="Jul 26" dateColor="bg-gray-300" />
      </ul>
    </div>
    <button className="mt-4 bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700">Continue Last Module</button>
  </div>
);

// Task Item Component
const TaskItem = ({ text, checked = false, date, dateColor }) => (
  <li className="flex items-center">
    <input type="checkbox" className="mr-2" checked={checked} readOnly />
    <span>{text}</span>
    {date && <span className={`ml-auto text-sm text-white ${dateColor} px-2 py-1 rounded`}>{date}</span>}
  </li>
);

// Team Dashboard Component
const TeamDashboard = () => (
  <div className="bg-white p-6 rounded-lg shadow-md w-1/2">
    <h2 className="text-lg font-bold mb-4">Team & HR Dashboard</h2>
    <table className="w-full mb-4">
      <thead>
        <tr>
          <th className="text-left font-semibold">Name</th>
          <th className="text-left font-semibold">Role</th>
          <th className="text-left font-semibold">Completion</th>
        </tr>
      </thead>
      <tbody>
        <TeamMember name="Jane Cooper" role="Marketing Coordinator" completion="75%" />
        <TeamMember name="Ronald Richards" role="Sales Associate" completion="Overdue" completionColor="text-red-600" />
        <TeamMember name="Cody Fisher" role="Product Designer" completion="60%" />
      </tbody>
    </table>
    <DashboardButton text="Assign Role-Specific Task" />
    <DashboardButton text="Send Reminder" />
    <DashboardButton text="Collect Feedback" />
  </div>
);

// Team Member Component
const TeamMember = ({ name, role, completion, completionColor = '' }) => (
  <tr>
    <td>{name}</td>
    <td>{role}</td>
    <td className={completionColor}>{completion}</td>
  </tr>
);

// Dashboard Button Component
const DashboardButton = ({ text }) => (
  <button className="w-full bg-white border py-2 rounded mb-2 hover:bg-gray-50">{text}</button>
);

// Main Dashboard Component
const Dashboard = () => {
  return (
    <div className="flex h-screen bg-gray-100">
      <Sidebar />
      <div className="flex-1 p-8">
        <Header />
        <div className="flex space-x-6">
          <OnboardingOverview />
          <TeamDashboard />
        </div>
      </div>
    </div>
  );
};

export default Dashboard;