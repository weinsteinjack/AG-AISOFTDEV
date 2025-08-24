import React from 'react';

const Dashboard = () => {
  return (
    <div className="flex h-screen bg-gray-100">
      {/* Sidebar */}
      <div className="w-20 bg-white border-r p-4 flex flex-col items-center">
        {/* Profile Icon */}
        <div className="bg-blue-500 p-3 rounded-full mb-4">
          {/* Profile SVG or icon here */}
        </div>
        {/* Sidebar Icons */}
        <nav className="flex flex-col items-center space-y-4 mt-4">
          {/* Icons placeholders */}
          <div className="w-8 h-8 bg-gray-300 rounded-full"></div>
          <div className="w-8 h-8 bg-gray-300 rounded-full"></div>
          <div className="w-8 h-8 bg-gray-300 rounded-full"></div>
        </nav>
      </div>

      {/* Main Content */}
      <div className="flex-1 p-8">
        {/* Header */}
        <header className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-bold">Welcome, Sarah!</h1>
          <div className="bg-white p-2 rounded-full">
            {/* Setting Icon */}
            {/* Settings SVG or icon here */}
          </div>
        </header>

        {/* Dashboard Content */}
        <div className="flex space-x-6">
          {/* My Onboarding Overview */}
          <div className="bg-white p-6 rounded-lg shadow-md w-1/2">
            <h2 className="text-lg font-bold mb-4">My Onboarding Overview</h2>
            <div className="flex items-center mb-4">
              {/* Progress Circle */}
              <div className="w-16 h-16 border-4 border-blue-500 rounded-full flex items-center justify-center">
                <span className="text-xl">40%</span>
              </div>
            </div>
            <div>
              <h3 className="font-semibold mb-2">Today's Tasks</h3>
              <ul className="space-y-2">
                <li className="flex items-center">
                  <input type="checkbox" className="mr-2" /> Complete Benefits Enrollment
                </li>
                <li className="flex items-center">
                  <input type="checkbox" className="mr-2" checked readOnly /> 
                  <span>Review Employee Handbook</span>
                  <span className="ml-auto text-sm text-white bg-yellow-500 px-2 py-1 rounded">Jul 26</span>
                </li>
                <li className="flex items-center">
                  <input type="checkbox" className="mr-2" /> 
                  <span>Set Up Workstation</span>
                  <span className="ml-auto text-sm text-white bg-gray-300 px-2 py-1 rounded">Jul 26</span>
                </li>
              </ul>
            </div>
            <button className="mt-4 bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700">Continue Last Module</button>
          </div>

          {/* Team & HR Dashboard */}
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
                <tr>
                  <td>Jane Cooper</td>
                  <td>Marketing Coordinator</td>
                  <td>75%</td>
                </tr>
                <tr>
                  <td>Ronald Richards</td>
                  <td>Sales Associate</td>
                  <td className="text-red-600">Overdue</td>
                </tr>
                <tr>
                  <td>Cody Fisher</td>
                  <td>Product Designer</td>
                  <td>60%</td>
                </tr>
              </tbody>
            </table>
            <button className="w-full bg-white border py-2 rounded mb-2 hover:bg-gray-50">Assign Role-Specific Task</button>
            <button className="w-full bg-white border py-2 rounded mb-2 hover:bg-gray-50">Send Reminder</button>
            <button className="w-full bg-white border py-2 rounded hover:bg-gray-50">Collect Feedback</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;