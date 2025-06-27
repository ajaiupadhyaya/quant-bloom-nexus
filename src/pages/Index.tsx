
import { ProfessionalTerminal } from "@/components/ProfessionalTerminal";
import { DashboardProvider } from "@/context/DashboardContext";

const Index = () => {
  return (
    <DashboardProvider>
      <ProfessionalTerminal />
    </DashboardProvider>
  );
};

export default Index;
